import numpy as np
import splinepy

rng = np.random.default_rng()
def r(a, b):
    return rng.uniform(a, b)

def f(base, p):
    return base * (1 + r(-p, p))

def random_geometry() -> splinepy.BSpline:
    """
    Creates a random BSpline geometry.

    Returns the created BSpline geometry.
    """
    b = 1
    h1 = 0.75
    h5 = h1

    p_b = 0.3
    p_0 = 1.4
    p_1 = 0.75
    p_2 = 0.75
    p_3 = 0.75
    p_4 = 0.75
    p_5 = 0.75


    bl = f(-b, p_b)
    br = f(b, p_b)
    h2 = f(-h1/2, 0)
    h3 = 0
    h4 = f(h1/2, 0)

    x1_r = f(br, p_1)
    x2_r = f(br, p_2)
    x3_r = f(br, p_3)
    x4_r = f(br, p_4)
    x5_r = f(br, p_5)

    x1_l = f(bl, p_1)
    x2_l = f(bl, p_2)
    x3_l = f(bl, p_3)
    x4_l = f(bl, p_4)
    x5_l = f(bl, p_5)


    cp = []

    cp.append([x1_r, -h1*p_0])
    cp.append([x1_r, -h1])
    cp.append([x2_r, h2])
    cp.append([x3_r, h3])
    cp.append([x4_r, h4])
    cp.append([x5_r, h5])
    cp.append([x5_r, h5*p_0])

    cp.append([x1_r * 0.666, -h1*p_0])
    cp.append([x1_r * 0.666, -h1])
    cp.append([x2_r * 0.666, h2])
    cp.append([x3_r * 0.666, h3])
    cp.append([x4_r * 0.666, h4])
    cp.append([x5_r * 0.666, h5])
    cp.append([x5_r * 0.666, h5*p_0])

    cp.append([x1_r * 0.333, -h1*p_0])
    cp.append([x1_r * 0.333, -h1])
    cp.append([x2_r * 0.333, h2])
    cp.append([x3_r * 0.333, h3])
    cp.append([x4_r * 0.333, h4])
    cp.append([x5_r * 0.333, h5])
    cp.append([x5_r * 0.333, h5*p_0])

    cp.append([x1_r * 0.0, -h1*p_0])
    cp.append([x1_r * 0.0, -h1])
    cp.append([x2_r * 0.0, h2])
    cp.append([x3_r * 0.0, h3])
    cp.append([x4_r * 0.0, h4])
    cp.append([x5_r * 0.0, h5])
    cp.append([x5_r * 0.0, h5*p_0])

    """
    cp.append([x1_l * 0.333, -h1*p_0])
    cp.append([x1_l * 0.333, -h1])
    cp.append([x2_l * 0.333, h2])
    cp.append([x3_l * 0.333, h3])
    cp.append([x4_l * 0.333, h4])
    cp.append([x5_l * 0.333, h5])
    cp.append([x5_l * 0.333, h5*p_0])

    cp.append([x1_l * 0.666, -h1*p_0])
    cp.append([x1_l * 0.666, -h1])
    cp.append([x2_l * 0.666, h2])
    cp.append([x3_l * 0.666, h3])
    cp.append([x4_l * 0.666, h4])
    cp.append([x5_l * 0.666, h5])
    cp.append([x5_l * 0.666, h5*p_0])

    cp.append([x1_l, -h1*p_0])
    cp.append([x1_l, -h1])
    cp.append([x2_l, h2])
    cp.append([x3_l, h3])
    cp.append([x4_l, h4])
    cp.append([x5_l, h5])
    cp.append([x5_l, h5*p_0])
    """

    cp.append([x1_l * 0.333, -h1*p_0])
    cp.append([x1_l * 0.333, -h1])
    cp.append([x2_l * 0.333, h2])
    cp.append([x3_l * 0.333, h3])
    cp.append([x4_l * 0.333, h4])
    cp.append([x5_l * 0.333, h5])
    cp.append([x5_l * 0.333, h5*p_0])

    cp.append([x1_l * 0.666, -h1*p_0])
    cp.append([x1_l * 0.666, -h1])
    cp.append([x2_l * 0.666, h2])
    cp.append([x3_l * 0.666, h3])
    cp.append([x4_l * 0.666, h4])
    cp.append([x5_l * 0.666, h5])
    cp.append([x5_l * 0.666, h5*p_0])

    cp.append([x1_l, -h1*p_0])
    cp.append([x1_l, -h1])
    cp.append([x2_l, h2])
    cp.append([x3_l, h3])
    cp.append([x4_l, h4])
    cp.append([x5_l, h5])
    cp.append([x5_l, h5*p_0])

    cp = sum(cp, [])
    cp = np.array(cp).reshape(-1, 2).tolist()

    knots = [[0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1], [0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1]]
    #knots = [[0, 0, 0, 0, 0, 0.333, 0.666, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0.333, 0.666, 1, 1, 1, 1, 1]]
    #knots = [[0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1], [0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1]]
    surf = splinepy.BSpline(degrees=[3, 3], knot_vectors=knots, control_points=cp)
    #surf.uniform_refine()
    return surf


if __name__ == "__main__":
    surf = random_geometry()
    splinepy.io.gismo.export("input_geometries/simple_random.xml", surf)