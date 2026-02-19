export crosscap_overlap

function crosscap_overlap(mps::MPS)
    N = mps.N
    @assert N % 4 == 0

    half_N = div(N, 2)
    A1 = mps.A[1][1, :, :]
    @tensor C[u, o, l] := A1[i, l] * mps.A[1+half_N][u, i, o]
    for n in 2:half_N-1
        @tensor C[u, y, p] := C[u, o, l] * mps.A[n][l, j, p] * mps.A[n+half_N][o, j, y]
    end
    AN = mps.A[N][:, :, 1]
    @tensor co = C[u, o, l] * mps.A[half_N][l, j, u] * AN[o, j]
    return co
end