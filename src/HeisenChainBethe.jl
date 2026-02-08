# Numerical solution of the Bethe Ansatz equations for the spin-1/2 Heisenberg chain
export heisen_chain_Bethe

function heisen_chain_Bethe(N::Int, BC::String)
    @assert BC == "OBC" || BC == "PBC"

    if BC == "OBC"
        E = N * (0.25 - log(2)) + (pi - 1 - 2 * log(2)) / 4
    else
        # set rapidity set
        BetheI = collect((1:N/2) .+ (-N / 4 - 1 / 2)) # Bethe quantum numbers for the ground state
        m = length(BetheI) # # of Bethe quantum numbers (= # of Bethe roots)

        u = copy(BetheI) # initial guess
        itnum = 1000 # number of iterations

        tempE = NaN
        for it1 in 1:itnum
            for counter in 1:m
                tempu = u[counter] .- u
                tempu = sum(atan.(tempu))
                u[counter] = 1 / 2 * tan((pi * BetheI[counter] + tempu) / N)
            end
            tempE = (N / 4 - 1 / 2 * sum(1 ./ (u .^ 2 .+ 1 / 4))) / N # per-site energy
        end
        E = tempE * N # converged energy
    end
    return E
end