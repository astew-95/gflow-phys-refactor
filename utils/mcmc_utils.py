def integrated_autocorr_time(x, c=5.0):
    """
    Integrated autocorrelation time of a 1D time series x, via Sokal's
    automatic-windowing estimator: tau_int(M) = 1 + 2*sum_{t=1}^{M} rho(t),
    where rho is the normalized autocorrelation function, and M is chosen
    as the smallest window where M >= c * tau_int(M) (c=5 is a standard
    default).
    Returns tau_int (float, >= 1.0).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 10:
        return float("nan")
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var < 1e-12:
        return float("nan")  # constant series (e.g., chain never moved)

    # autocorrelation via FFT (much faster than direct summation for long chains)
    size = 1
    while size < 2 * n:
        size *= 2
    fx = np.fft.rfft(x, n=size)
    acf = np.fft.irfft(fx * np.conjugate(fx), n=size)[:n].real
    acf /= (var * n)  # normalize so acf[0] == 1

    tau = 1.0
    for M in range(1, n // 2):
        tau = 1.0 + 2.0 * np.sum(acf[1:M + 1])
        if M >= c * tau:
            return max(tau, 1.0)
    return max(tau, 1.0)  # window never satisfied c*tau <= M; best available estimate