import pandas as pd

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].x_values
        y = points.iloc[i].y_values

        error = y - (m_now * x + b_now)
        m_gradient += -(2/n) * x * error
        b_gradient += -(2/n) * error

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b
