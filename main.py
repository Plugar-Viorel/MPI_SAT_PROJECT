import random
import time
import tracemalloc
import matplotlib.pyplot as plt
from itertools import combinations
from typing import List, Set, Dict, Tuple, Optional

# -----------------------
# Tipuri personalizate
# -----------------------
Literal = int                       # Reprezintă un literal (variabilă sau negarea ei)
Clause = Set[Literal]               # O clauză este un set de literali
CNF    = List[Clause]               # Formula CNF: listă de clauze
Stats  = Dict[str, int]             # Statistici colectate de solvers

# -----------------------
# Generator CNF
# -----------------------
# Creează o instanță CNF aleatoare
#   num_vars   : numărul maxim de variabile (literalii vor fi în 1..num_vars)
#   num_clauses: câte clauze să conțină formula
#   width      : lungimea fiecărei clauze (dificultatea crește cu width și num_clauses)
# Pentru a modifica dificultatea instanțelor, ajustează: num_vars, num_clauses și width.

def create_cnf(num_vars: int, num_clauses: int, width: int = 3) -> CNF:
    formula, seen = [], set()
    removed, attempts = 0, 0
    # Limita de încercări previne bucle infinite când filtrăm duplicate
    max_attempts = num_clauses * 8
    while len(formula) < num_clauses and attempts < max_attempts:
        clause = set()
        # Construim o clauză de lungime 'width'
        while len(clause) < width:
            v = random.randint(1, num_vars)
            lit = v if random.choice([True, False]) else -v
            clause.add(lit)
        attempts += 1
        # Filtrăm clauze triviale sau duplicate
        if any(-l in clause for l in clause): removed += 1; continue
        key = frozenset(clause)
        if key in seen: removed += 1; continue
        formula.append(clause); seen.add(key)
    print(f"[Generator] Instanță: {len(formula)} clauze, {removed} filtrate")
    return formula

# -----------------------
# Simplificare CNF
# -----------------------
# După asignarea unui literal:
#   - Elimină orice clauză care conține literalul asignat (clauza e satisfăcută)
#   - Elimină negarea literalului din celelalte clauze
#   - Dacă vreo clauză devine vidă, returnează None (contradicție)
def simplify(cnf: CNF, literal: Literal) -> Optional[CNF]:
    next_cnf = []
    for clause in cnf:
        if literal in clause:
            # Clauza satisfăcută, o eliminăm
            continue
        # Eliminăm literalii contraziși
        reduced = clause - {-literal}
        if not reduced:
            # Clauză vidă -> conflict
            return None
        next_cnf.append(reduced)
    return next_cnf

# =======================
# Solvers de bază
# =======================
class BaseSolver:
    """
    Clasa de bază pentru toți solverii.
    Definește interfața solve(cnf) -> (sat, stats)
    """
    def solve(self, cnf: CNF) -> Tuple[bool, Stats]:
        raise NotImplementedError

class ResolutionSolver(BaseSolver):
    """
    Aplică rezoluție exhaustivă până la punct fix sau până găsește clauza vidă.
    Contorizează fiecare rezolvent ca decizie și fiecare detectare a vidului ca backtrack.
    """
    def solve(self, cnf: CNF) -> Tuple[bool, Stats]:
        stats = {'decisions': 0, 'backtracks': 0}
        clauses = {frozenset(c) for c in cnf}
        while True:
            new_res = set()
            for c1, c2 in combinations(clauses, 2):
                for lit in c1:
                    if -lit in c2:
                        stats['decisions'] += 1
                        rez = (c1 | c2) - {lit, -lit}
                        if not rez:
                            stats['backtracks'] += 1
                            return False, stats
                        new_res.add(frozenset(rez))
            # Dacă nu apare nimic nou, formula este satisfiabilă
            if not new_res or new_res.issubset(clauses):
                return True, stats
            clauses |= new_res

# =======================
# DPLL extins
# =======================
class DPLLSolver(BaseSolver):
    """
    Algoritmul DPLL cu:
      - propagare unitară
      - eliminare literali puri
      - ramificare cu diverse euristici (SPL, RND, MOMS, JW)
    """
    def __init__(self, heuristic: str = 'spl'):
        self.stats = {'decisions': 0, 'propagations': 0, 'backtracks': 0}
        self.heuristic = heuristic  # strategia de ramificare

    def unit_prop(self, cnf: CNF, assign: Dict[int,bool]) -> Tuple[Optional[CNF], Dict[int,bool]]:
        """
        Aplică propagare unitară: pentru fiecare clauză de lungime 1,
        asignăm literalul și simplificăm recursiv.
        Dacă apare conflict, semnalăm backtrack.
        """
        changed = True
        while changed:
            changed = False
            for cl in list(cnf):
                if len(cl) == 1:
                    lit = next(iter(cl)); var = abs(lit)
                    if var not in assign:
                        assign[var] = (lit > 0)
                        self.stats['propagations'] += 1
                        cnf = simplify(cnf, lit)
                        if cnf is None:
                            self.stats['backtracks'] += 1
                            return None, assign
                        changed = True
                        break
        return cnf, assign

    def pure_lit(self, cnf: CNF, assign: Dict[int,bool]) -> Tuple[Optional[CNF], Dict[int,bool]]:
        """
        Elimină literalii puri: apar doar cu un semn în toate clauzele,
        îi asignăm direct și simplificăm.
        """
        freq = {}
        for cl in cnf:
            for l in cl:
                freq[l] = freq.get(l, 0) + 1
        for l in list(freq):
            if -l not in freq:
                var = abs(l)
                if var not in assign:
                    assign[var] = (l > 0)
                    cnf = simplify(cnf, l)
                    if cnf is None:
                        self.stats['backtracks'] += 1
                        return None, assign
        return cnf, assign

    # =====================
    # Euristici(Strategii) de ramificare
    # =====================
    def pick_var_spl(self, cnf, assign):
        """SPL: alege literal din clauza cu lungimea minimă"""
        cl = min((cl for cl in cnf if any(abs(l) not in assign for l in cl)), key=len)
        return next(l for l in cl if abs(l) not in assign)

    def pick_var_rand(self, cnf, assign):
        """RND: alege aleatoriu o variabilă neatribuită"""
        unassigned = {abs(l) for cl in cnf for l in cl if abs(l) not in assign}
        return random.choice(list(unassigned))

    def pick_var_moms(self, cnf, assign):
        """MOMS: maximizează aparițiile în clauzele de dimensiune minimă"""
        min_size = min(len(cl) for cl in cnf if any(abs(l) not in assign for l in cl))
        counts = {}
        for cl in cnf:
            if len(cl) == min_size:
                for l in cl:
                    v = abs(l)
                    if v not in assign:
                        counts[v] = counts.get(v, 0) + 1
        return max(counts, key=counts.get)

    def pick_var_jw(self, cnf, assign):
        """Jeroslow–Wang: ponderare cu 2^-|clauză|"""
        scores = {}
        for cl in cnf:
            w = 2 ** -len(cl)
            for l in cl:
                v = abs(l)
                if v not in assign:
                    scores[v] = scores.get(v, 0) + w
        return max(scores, key=scores.get)

    def pick_var(self, cnf: CNF, assign: Dict[int,bool]) -> int:
        """
        Înregistrează decizia și aplică euristica selectată.
        """
        self.stats['decisions'] += 1
        if self.heuristic == 'rnd': return self.pick_var_rand(cnf, assign)
        if self.heuristic == 'moms': return self.pick_var_moms(cnf, assign)
        if self.heuristic == 'jw':  return self.pick_var_jw(cnf, assign)
        return self.pick_var_spl(cnf, assign)

    def solve(self, cnf: CNF) -> Tuple[bool, Stats]:
        """
        Pornim DFS cu propagare unitară, eliminare puri și ramificare
        până găsim satisfacibilitate sau eșuăm (backtrack).
        """
        def dfs(curr_cnf: CNF, assignment: Dict[int,bool]) -> bool:
            curr_cnf, assignment = self.unit_prop(curr_cnf, assignment)
            if curr_cnf is None: return False
            if not curr_cnf: return True
            curr_cnf, assignment = self.pure_lit(curr_cnf, assignment)
            if curr_cnf is None: return False
            if not curr_cnf: return True
            var = self.pick_var(curr_cnf, assignment)
            for val in [True, False]:
                new_assign = assignment.copy()
                new_assign[var] = val
                reduced = simplify(curr_cnf, var if val else -var)
                if reduced and dfs(reduced, new_assign):
                    return True
            self.stats['backtracks'] += 1
            return False
        sat = dfs(cnf, {})
        return sat, self.stats

class DPMSolver(BaseSolver):
    """
    Algoritmul Davis–Putnam original: elimină variabile în ordine fixă,
    generează rezolvenți pentru fiecare variabilă.
    """
    def solve(self, cnf: CNF) -> Tuple[bool, Stats]:
        stats = {'decisions': 0, 'backtracks': 0}
        vars_all = {abs(l) for cl in cnf for l in cl}
        # Ordine descrescătoare după frecvența variabilelor
        order = sorted(vars_all, key=lambda v: -sum((v in c or -v in c) for c in cnf))
        for v in order:
            stats['decisions'] += 1
            pos = [c for c in cnf if v in c]
            neg = [c for c in cnf if -v in c]
            rest = [c for c in cnf if v not in c and -v not in c]
            new_res = []
            for p in pos:
                for n in neg:
                    rez = (p | n) - {v, -v}
                    if not rez:
                        stats['backtracks'] += 1
                        return False, stats
                    new_res.append(rez)
            cnf = rest + new_res
        return True, stats

# =======================
# Experiment comparativ
# =======================
class Experiment:
    """
    Rulează toți solverii și raportează:
     - dimensiunea instanței (nr. de clauze)
     - timpul de execuție, memoria, deciziile și backtrack-urile
    Ajustează:
      * vars_n (num_vars)
      * clause_sizes (lista cu mărimi de instanțe)
    """
    def __init__(self, vars_n=6, clause_sizes=None):
        self.vars_n = vars_n
        self.clause_sizes = clause_sizes or list(range(10,16,2))
        self.solvers = {
            'Rezoluție': ResolutionSolver(),
            'Davis-Putnam': DPMSolver(),
            'DPLL (SPL)': DPLLSolver('spl'),
            'DPLL (RND)': DPLLSolver('rnd'),
            'DPLL (MOMS)': DPLLSolver('moms'),
            'DPLL (JW)':  DPLLSolver('jw')
        }

    def run(self) -> Dict[str, Dict[str, List]]:
        results = {name: {'size': [], 'time': [], 'mem': [], 'sat': [], 'decisions': [], 'backtracks': []}
                   for name in self.solvers}
        for sz in self.clause_sizes:
            print(f"--- {sz} clauze ---")
            cnf = create_cnf(self.vars_n, sz)
            for name, solver in self.solvers.items():
                if name == 'Rezoluție' and sz > 12: continue
                copy_cnf = [set(c) for c in cnf]
                tracemalloc.start(); t0 = time.perf_counter()
                sat, stats = solver.solve(copy_cnf)
                dt = time.perf_counter() - t0; _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
                res = results[name]
                res['size'].append(sz); res['time'].append(dt)
                res['mem'].append(peak/1024); res['sat'].append(sat)
                res['decisions'].append(stats.get('decisions', 0))
                res['backtracks'].append(stats.get('backtracks', 0))
        return results

    def report(self, data):
        header = ['Solver','Clauze','SAT','Timp(s)','Mem(KB)','Decizii','Backtracks']
        rows = []
        for name, stats in data.items():
            for i, sz in enumerate(stats['size']):
                rows.append([name, sz,
                             '✓' if stats['sat'][i] else '✗',
                             f"{stats['time'][i]:.3f}",
                             f"{stats['mem'][i]:.1f}",
                             stats['decisions'][i],
                             stats['backtracks'][i]])
        # Afișare tabel și grafic
        widths = [max(len(str(r[i])) for r in ([header]+rows)) for i in range(len(header))]
        fmt = ' | '.join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*header)); print('-'*(sum(widths)+3*(len(widths)-1)))
        for r in rows: print(fmt.format(*r))
        fig, ax = plt.subplots(figsize=(len(header)*1.2, len(rows)*0.4+1))
        ax.set_title("Proiect SAT: analiză comparativă", pad=20)
        ax.axis('off')
        table_data = [header] + [[str(c) for c in row] for row in rows]
        table = ax.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1,1.5)
        plt.show()

if __name__ == '__main__':
    # Exemplu de configurare: modifică vars_n și clause_sizes pentru instanțe mai dificile
    exp = Experiment(vars_n=9, clause_sizes=[6,8,10,12])
    results = exp.run()
    exp.report(results)
