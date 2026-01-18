"""
earth_system_simulation.py

Minimal Earth System Model example illustrating:
- object-oriented design
- state variables
- time stepping
- component coupling
- clean diagnostic output

Author: Roland Potthast (example code)
"""

# =============================================================================
# Base class
# =============================================================================
class EarthSystemComponent:
    def __init__(self, name):
        self.name = name

    def step(self, dt, forcing=None):
        raise NotImplementedError("This method must be implemented by subclasses")


# =============================================================================
# Atmosphere
# =============================================================================
class Atmosphere(EarthSystemComponent):
    def __init__(self, name, temperature=288.0):
        super().__init__(name)
        self.temperature = temperature  # Kelvin
        print(f"[INIT] Atmosphere '{self.name}' with T={self.temperature:.2f} K")

    def step(self, dt, forcing=None):
        sst = forcing.get("sst", self.temperature)
        # simple relaxation toward sea surface temperature
        self.temperature += dt * 0.01 * (sst - self.temperature)


# =============================================================================
# Ocean
# =============================================================================
class Ocean(EarthSystemComponent):
    def __init__(self, name, sst=290.0):
        super().__init__(name)
        self.sst = sst  # Kelvin
        print(f"[INIT] Ocean '{self.name}' with SST={self.sst:.2f} K")

    def step(self, dt, forcing=None):
        atm_temp = forcing.get("atm_temp", self.sst)
        # weak coupling toward atmospheric temperature
        self.sst += dt * 0.005 * (atm_temp - self.sst)


# =============================================================================
# Land
# =============================================================================
class Land(EarthSystemComponent):
    def __init__(self, name, soil_moisture=0.30):
        super().__init__(name)
        self.soil_moisture = soil_moisture  # fraction
        print(f"[INIT] Land '{self.name}' with soil moisture={self.soil_moisture:.2f}")

    def step(self, dt, forcing=None):
        # simple drying process
        self.soil_moisture = max(self.soil_moisture - dt * 0.001, 0.0)


# =============================================================================
# Earth System Model
# =============================================================================
class EarthSystemModel:
    def __init__(self, atmosphere, ocean, land):
        self.atmosphere = atmosphere
        self.ocean = ocean
        self.land = land
        self.time = 0.0
        print("[INIT] Earth System Model initialized")

    def step(self, dt):
        print(
            f"[STEP] Advancing from t={self.time:.1f} to t={self.time + dt:.1f} | "
            f"forcing: ATM<-SST={self.ocean.sst:.2f}, "
            f"OCN<-T_atm={self.atmosphere.temperature:.2f}"
        )

        # component coupling and time stepping
        self.atmosphere.step(dt, forcing={"sst": self.ocean.sst})
        self.ocean.step(dt, forcing={"atm_temp": self.atmosphere.temperature})
        self.land.step(dt)

        self.time += dt

    def run(self, nsteps, dt):
        print("[RUN] Starting simulation\n")
        for _ in range(nsteps):
            self.step(dt)
            print(
                f"t={self.time:5.1f}  "
                f"T_atm={self.atmosphere.temperature:6.2f} K  "
                f"SST={self.ocean.sst:6.2f} K  "
                f"Soil={self.land.soil_moisture:4.2f}"
            )
        print("\n[RUN] Simulation finished")


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    atmosphere = Atmosphere("Global Atmosphere")
    ocean = Ocean("Global Ocean")
    land = Land("Global Land")

    model = EarthSystemModel(atmosphere, ocean, land)
    model.run(nsteps=20, dt=1.0)
