import numpy as np
import matplotlib.pyplot as plt
import csv

# =========================
# Physical constants
# =========================
M_PROTON = 0.938272  # GeV


# =========================
# Relativistic kinematics
# =========================
def com_energy_from_sqrt_s(sqrt_s):
    """
    Returns single-particle energy in the COM frame.
    For p + p -> p + p, initial particles have:
        E = sqrt(s) / 2
    """
    return sqrt_s / 2.0


def com_momentum_from_sqrt_s(sqrt_s, mass=M_PROTON):
    """
    Returns momentum magnitude in the COM frame:
        p = sqrt(E^2 - m^2)
    """
    E = com_energy_from_sqrt_s(sqrt_s)
    p2 = E**2 - mass**2
    if p2 < 0:
        raise ValueError("sqrt(s) is below the elastic pp threshold.")
    return np.sqrt(p2)


def mandelstam_t(p, cos_theta):
    """
    In COM frame for elastic scattering:
        t = -2 p^2 (1 - cos(theta))
    """
    return -2.0 * p**2 * (1.0 - cos_theta)


def mandelstam_u(p, cos_theta):
    """
    In COM frame for elastic scattering:
        u = -2 p^2 (1 + cos(theta))
    """
    return -2.0 * p**2 * (1.0 + cos_theta)


# =========================
# Cross-section models
# =========================
def dsigma_dt_exponential(t, B=8.0):
    """
    Toy hadronic forward-peaked model:
        dσ/dt ∝ exp(B t)

    Since t <= 0, this favors small |t|, i.e. forward scattering.
    B is in GeV^{-2} in a rough phenomenological sense.
    """
    return np.exp(B * t)


def dsigma_domega_isotropic():
    """
    Simplest possible model:
        dσ/dΩ = constant
    """
    return 1.0


def event_weight_forward_peaked(p, cos_theta, B=8.0):
    """
    Weight proportional to dσ/dt evaluated at t(cosθ).
    This is a toy model for sampling purposes.
    """
    t = mandelstam_t(p, cos_theta)
    return dsigma_dt_exponential(t, B=B)


# =========================
# Event construction
# =========================
def build_event(sqrt_s, cos_theta, phi, mass=M_PROTON):
    """
    Constructs the full elastic pp event in the COM frame.

    Initial state:
        p1 = (E, 0, 0, +p)
        p2 = (E, 0, 0, -p)

    Final state:
        p3 = (E, px, py, pz)
        p4 = (E, -px, -py, -pz)
    """
    E = com_energy_from_sqrt_s(sqrt_s)
    p = com_momentum_from_sqrt_s(sqrt_s, mass=mass)

    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta**2))

    px = p * sin_theta * np.cos(phi)
    py = p * sin_theta * np.sin(phi)
    pz = p * cos_theta

    p1 = np.array([E, 0.0, 0.0, +p])
    p2 = np.array([E, 0.0, 0.0, -p])
    p3 = np.array([E, +px, +py, +pz])
    p4 = np.array([E, -px, -py, -pz])

    t = mandelstam_t(p, cos_theta)
    u = mandelstam_u(p, cos_theta)
    s = sqrt_s**2

    return {
        "sqrt_s": sqrt_s,
        "s": s,
        "t": t,
        "u": u,
        "theta": np.arccos(cos_theta),
        "phi": phi,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
    }


# =========================
# Monte Carlo generator
# =========================
def generate_events(n_events, sqrt_s, B=8.0, max_trials=10_000_000):
    """
    Acceptance-rejection event generator.

    Proposal distribution:
        cos(theta) uniformly in [-1, 1]
        phi uniformly in [0, 2π)

    Target weight:
        w(cosθ) ∝ exp(B t(cosθ))
    """
    p = com_momentum_from_sqrt_s(sqrt_s)

    # Maximum weight occurs at t = 0 => exp(B*0)=1
    w_max = 1.0

    accepted_events = []
    n_trials = 0

    while len(accepted_events) < n_events and n_trials < max_trials:
        n_trials += 1

        cos_theta = np.random.uniform(-1.0, 1.0)
        phi = np.random.uniform(0.0, 2.0 * np.pi)

        w = event_weight_forward_peaked(p, cos_theta, B=B)
        r = np.random.uniform(0.0, w_max)

        if r < w:
            event = build_event(sqrt_s, cos_theta, phi)
            accepted_events.append(event)

    if len(accepted_events) < n_events:
        raise RuntimeError(
            f"Only generated {len(accepted_events)} events after {max_trials} trials."
        )

    efficiency = len(accepted_events) / n_trials
    return accepted_events, efficiency


# =========================
# Output
# =========================
def save_events_to_csv(events, filename="events.csv"):
    """
    Saves a flat table of event-level quantities.
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sqrt_s", "s", "t", "u", "theta", "phi",
            "p3_E", "p3_px", "p3_py", "p3_pz",
            "p4_E", "p4_px", "p4_py", "p4_pz"
        ])

        for ev in events:
            p3 = ev["p3"]
            p4 = ev["p4"]
            writer.writerow([
                ev["sqrt_s"], ev["s"], ev["t"], ev["u"], ev["theta"], ev["phi"],
                p3[0], p3[1], p3[2], p3[3],
                p4[0], p4[1], p4[2], p4[3]
            ])


def plot_results(events):
    theta = np.array([ev["theta"] for ev in events])
    t_vals = np.array([ev["t"] for ev in events])
    pz_vals = np.array([ev["p3"][3] for ev in events])

    plt.figure(figsize=(8, 5))
    plt.hist(theta, bins=60)
    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel("Counts")
    plt.title("Elastic pp scattering: angular distribution")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(t_vals, bins=60)
    plt.xlabel(r"$t\ \mathrm{[GeV^2]}$")
    plt.ylabel("Counts")
    plt.title("Elastic pp scattering: Mandelstam t distribution")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(pz_vals, bins=60)
    plt.xlabel(r"$p_{z,3}\ \mathrm{[GeV]}$")
    plt.ylabel("Counts")
    plt.title("Final proton longitudinal momentum")
    plt.tight_layout()
    plt.show()


# =========================
# Main
# =========================
if __name__ == "__main__":
    np.random.seed(42)

    sqrt_s = 5.0   # GeV
    n_events = 20000
    B = 8.0        # forward-peaking slope parameter

    events, efficiency = generate_events(n_events=n_events, sqrt_s=sqrt_s, B=B)

    print(f"Generated {len(events)} events")
    print(f"Acceptance efficiency = {efficiency:.4f}")

    save_events_to_csv(events, "events.csv")
    plot_results(events)
