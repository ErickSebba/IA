from experta import KnowledgeEngine, Fact, Rule, AS, OR, NOT
import random

class Perception(Fact):
    pass

class Safe(Fact):
    pass

class WumpusWorld(KnowledgeEngine):

    def __init__(self, grid_size=4, random=True):

        super().__init__()
        self.visited = set()
        self.grid_size = grid_size
        self.player_pos = (0, 0)
        self.possible_W = set()
        self.possible_H = set()

        if random:
            self.wumpus_pos = (random.randint(1, grid_size-1), random.randint(1, grid_size-1))
            self.gold_pos = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))

            while self.gold_pos == self.wumpus_pos:
                self.gold_pos = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))

            self.holes = set()
            for _ in range(random.randint(1, grid_size-1)):
                hole_pos = (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
                if hole_pos not in {self.wumpus_pos, self.gold_pos, self.player_pos}:
                    self.holes.add(hole_pos)
        else:

            '''self.gold_pos = (1, 1)
            self.wumpus_pos = (3, 1)
            self.holes = {(2, 2), (2, 3)}'''
            self.gold_pos = (2, 1)
            self.wumpus_pos = (2, 0)
            self.holes = {(0, 2), (2, 2)}


    def adjacent_positions(self):
        return [
            (self.player_pos[0] + dx, self.player_pos[1] + dy)
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
            if 0 <= self.player_pos[0] + dx < self.grid_size and 0 <= self.player_pos[1] + dy < self.grid_size
        ]

    def start(self):
        """Inicializa o agente e ativa a inferência."""
        self.display_board()
        self.declare(Safe(pos=self.player_pos))
        self.check_environment()
        self.run()

    def display_board(self):
        """Exibe o tabuleiro."""
        board = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        board[self.gold_pos[1]][self.gold_pos[0]] = 'G'
        board[self.wumpus_pos[1]][self.wumpus_pos[0]] = 'W'
        for hole in self.holes:
            board[hole[1]][hole[0]] = 'H'
        board[self.player_pos[1]][self.player_pos[0]] = 'P'
        print("\n".join(" ".join(row) for row in board))
        print()

    '''@Rule(AS.p << Perception(brisa=True) | AS.p << Perception(fedor=True))
    def move_prev_pos(self, p):
        """Se encontrar perigo, vai para outra posicao do segura."""
        self.retract(p)
        print("Há brisa ou fedor.")
        self.visited.add(self.player_pos)
        self.decide_move()'''

    @Rule(AS.p << Perception(brisa=False, fedor=False))
    def mark_safe_cells(self, p):
        """Marca células seguras ao redor do jogador se não houver perigo."""
        print("Não há brisa nem fedor. Marcando células adjacentes como seguras.")
        self.retract(p)
        self.visited.add(self.player_pos)

        for pos in self.adjacent_positions():
            self.declare(Safe(pos=pos))
            print(f"Célula {pos} marcada como segura.")

        self.decide_move()

    @Rule(AS.p << Perception(brisa=True, fedor=False))
    def avoid_hole(self, p):
        """Se há brisa, um buraco está próximo. Evita as células ao redor."""
        print("Há brisa! Um buraco está por perto. Evitando células perigosas.")
        self.retract(p)
        self.visited.add(self.player_pos)

        for pos in self.adjacent_positions():
            if pos not in self.visited:
                if pos in self.possible_W:  # Se já era possível Wumpus, agora sabemos que é segura!
                    print(f"{pos} era possível Wumpus e possível Buraco, então agora é segura!")
                    self.declare(Safe(pos=pos))
                else:
                    self.possible_H.add(pos)
                    print(f"Marcando {pos} como possível buraco.")

        self.decide_move()


    @Rule(AS.p << Perception(brisa=False, fedor=True))
    def avoid_wumpus(self, p):
        """Se há fedor, o Wumpus está próximo. Evita as células ao redor."""
        print("Há fedor! O Wumpus está por perto. Evitando possíveis posições do Wumpus.")
        self.retract(p)
        self.visited.add(self.player_pos)

        for pos in self.adjacent_positions():
            if pos not in self.visited:
                if pos in self.possible_H:  # Se já era possível Buraco, agora sabemos que é segura!
                    print(f"{pos} era possível Wumpus e possível Buraco, então agora é segura!")
                    self.declare(Safe(pos=pos))
                else:
                    self.possible_W.add(pos)
                    print(f"Marcando {pos} como possível Wumpus.")

        self.decide_move()

    '''@Rule(AS.p << Perception(possible_W=True, possible_H=True)) #enrtender como fazer funcionar
    def mark_safe_cells(self, p):
        """Marca células seguras ao redor do jogador se não houver perigo."""
        print("Não há brisa nem fedor. Marcando células adjacentes como seguras.")
        self.retract(p)
        self.visited.add(self.player_pos)

        for pos in self.adjacent_positions():
            self.declare(Safe(pos=pos))
            print(f"Célula {pos} marcada como segura.")

        self.decide_move()'''

    def decide_move(self):
        """Decide o próximo movimento."""
        print("Procurando movimento")

        safe_cells = {fact["pos"] for fact_id, fact in self.facts.items() if isinstance(fact, Safe)}
        unvisited_safe_cells = [cell for cell in safe_cells if cell not in self.visited]

        print("Celulas nao visitadas: ", unvisited_safe_cells)

        if unvisited_safe_cells:
            move = random.choice(unvisited_safe_cells)
        else:
            print("O agente está preso e não pode se mover. Fim de jogo.")
            exit()

        self.execute_move(move)

    def execute_move(self, new_pos):
        """Executa o movimento e atualiza o ambiente."""
        self.visited.add(self.player_pos)
        self.player_pos = new_pos

        print(f"Agente moveu para {self.player_pos}")
        self.display_board()

        self.check_environment()

        self.run()

    def check_death(self):
        """Verifica se o agente caiu em um buraco ou foi pego pelo Wumpus."""

        if self.player_pos in self.holes:
            print("O agente caiu em um buraco! Fim de jogo.")
            exit()
        if self.player_pos == self.wumpus_pos:
            print("O agente foi pego pelo Wumpus! Fim de jogo.")
            exit()

    def found_gold(self):
        """Verifica se o agente encontrou o ouro"""
        if self.player_pos == self.gold_pos:
            print("O agente encontrou o ouro! Fim de jogo!")
            exit()

    def check_environment(self):
        """Verifica o ambiente e atualiza percepções."""
        print("Verificando o ambiente...")

        self.found_gold()

        self.check_death()

        adjacent_positions = self.adjacent_positions()

        brisa = any(pos in self.holes for pos in adjacent_positions)
        fedor = any(pos == self.wumpus_pos for pos in adjacent_positions)
        brilho = self.player_pos == self.gold_pos

        print("Declarando as percepcoes ", brisa, fedor, brilho)
        self.declare(Perception(brisa=brisa, fedor=fedor, brilho=brilho))



# Inicializa o motor
try:
    engine = WumpusWorld(random=False)
    engine.reset()
    engine.start()
except:
    print("FIM")