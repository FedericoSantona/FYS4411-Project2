
class Particle:
    def __init__(self, position, mass=1.0):
        self.position = position  # Position can be a NumPy array or a JAX array, depending on backend
        self.mass = mass


class Boson(Particle):
    def __init__(self, position, mass=1.0):
        super().__init__(position, mass)
        self.type = 'boson'

class Fermion(Particle):
    def __init__(self, position, mass=1.0):
        super().__init__(position, mass)
        self.type = 'fermion'
