# Woven Fabric Unit Cell Optimization
# This script optimizes the geometry of a woven fabric unit cell using Euler elastica energy minimization
using LinearAlgebra
using Optim
using Roots
using QuadGK
using DelimitedFiles
import GLMakie: Axis, Axis3, Figure, scatter!, linesegments!, save, labelslidergrid!, Label, Box, set_close_to!, LScene, cam3d!, @lift, mesh, lines!
using DataFrames
using Dates
using CSV
using ForwardDiff

# Numerical inverse function using root finding
function numerical_inverse(f, y_target; x_guess=0.0, tol=1e-10)
    """
    Computes the numerical inverse of a function at a given point.
    """
    g(x) = f(x) - y_target
    x = find_zero(g, x_guess; atol=tol)
    return x
end

# Calculate angle between two vectors
function angle(a, b)
    return sqrt((a' * b)^2 / (a' * a * b' * b))
end

# Calculate lengths of cable segments
function calclencables(xs)
    """Calculate lengths between consecutive points"""
    lenths = [norm(xs[i+1, :] - xs[i, :]) for i in 1:size(xs, 1)-1]
    return lenths
end

# Calculate angles between consecutive cable segments
function calcanglecables(xs)
    """Calculate angles between consecutive segments"""
    listdif = [xs[i+1, :] - xs[i, :] for i in 1:size(xs)[1]-1]
    angles = [angle(listdif[i+1], listdif[i]) for i in 1:size(xs)[1]-2]
    return angles
end

# Calculate fiddle cable lengths
function calclenfidcables(xs)
    """Calculate lengths skipping one point"""
    lenths = [norm(xs[i-1, :] - xs[i+1, :]) for i in 2:size(xs)[1]-1]
    return lenths
end

# Initialize a sinusoidal thread
function initializeThread_sin(n, A, w, ϕ)
    """
    Initialize a thread with sinusoidal shape
    n: number of points
    A: amplitude
    w: frequency
    ϕ: phase shift
    """
    p0 = Float64[]
    push!(p0, 0.0)  # Start point
    
    # Calculate arc length
    f(s) = sqrt(1 + A^2 * (pi * w)^2 * cos(pi * w * s + ϕ)^2)
    len = (quadgk(f, 0, 1 / w)[1])
    
    # Distribute points along arc length
    for i in 2:n-1
        s(t) = quadgk(f, 0, t)[1]
        x_inverse = numerical_inverse(s, len * (i - 1) / (n - 1), x_guess=0.1)
        push!(p0, x_inverse)
        push!(p0, 0)
        push!(p0, A * sin(pi * w * x_inverse + ϕ))
    end
    push!(p0, 1 / w)
    
    p0 = Float64.(p0)
    ds = 2 * len / (n - 1)
    
    # Create cable connectivity
    cables = []
    for i in 1:n-1
        push!(cables, [i, i + 1])
    end
    
    return cables, p0, len, ds
end

# Initialize the unit cell with multiple threads
function initializeUnitCell(number_of_threads, A, wR, wC, n)
    """
    Initialize woven fabric unit cell
    wR: row frequency
    wC: column frequency
    """
    threads = []
    dsR = 0.0
    dsC = 0.0
    
    # Initialize row threads
    ϕ = 0
    shift = (2 / wC) / 4
    cables, p0, len, dsR = initializeThread_sin(n, A, wR, ϕ)
    for j in 2+1:3:length(p0)-1
        p0[j] += shift
    end
    push!(threads, p0)
    
    # Initialize column threads
    ϕ = pi
    shift = (2 / wR) / 4
    cables, p0, len, dsC = initializeThread_sin(n, A, wC, ϕ)
    q0 = Float64[]
    push!(threads, p0[1])
    for i in 1+1:3:length(p0)-1
        push!(q0, p0[i+1] + shift)
        push!(q0, p0[i])
        push!(q0, p0[i+2])
    end
    push!(threads, q0)
    push!(threads, p0[end])
    
    return threads, dsR, dsC
end

# Add boundary points to threads
function addBoundarytoThreats(threads, wR, wC)
    """Add boundary points to thread endpoints"""
    threadsboundaries = []
    for i in 1:size(threads)[1]
        if i == 1
            shift = (2 / wC) / 4
            vec = vcat([threads[i][1], 0.0 + shift, 0.0], threads[i][2:end-1], [threads[i][end], 0.0 + shift, 0.0])
        end
        if i == 2
            shift = (2 / wR) / 4
            vec = vcat([0.0 + shift, threads[i][1], 0.0], threads[i][2:end-1], [0.0 + shift, threads[i][end], 0.0])
        end
        push!(threadsboundaries, vec)
    end
    return threadsboundaries
end

# Visualization function
function plot_unit_cell!(ax, threads, n, wR, wC)
    """Plot the unit cell structure"""
    threads = [collect(part) for part in Iterators.partition(threads, 3 * (n - 2) + 2)]
    threads = addBoundarytoThreats(threads, wR, wC)
    
    for thread in threads
        points = map(i -> (Float32(thread[i]), Float32(thread[i+1]), Float32(thread[i+2])), 1:3:length(thread))
        scatter!(ax, points, markersize=5, color=:blue)
        lines!(ax, points, color=:blue)
    end
end

# Energy calculation function
function euler_elastica_energy(threads, n, wR, wC, dsR, dsC, r, l0r, l0c, kb)
    """
    Calculate total elastic energy including stretching, bending, and repulsion
    kb: bending stiffness
    r: thread radius
    l0r, l0c: rest lengths
    """
    threads = [collect(part) for part in Iterators.partition(threads, 3 * (n - 2) + 2)]
    threads = addBoundarytoThreats(threads, wR, wC)
    
    E = 0.0
    Estretch = 0.0
    Ebending = 0.0
    Ebars = 0.0
    
    # Calculate stretching and bending energy for each thread
    for j in 1:length(threads)
        thread = threads[j]
        ds = (j % 2 == 1) ? l0r / (n - 1) : l0c / (n - 1)
        qpoints = reshape(thread, (3, n))'
        lens = calclencables(qpoints)
        
        # Stretching energy
        for i in 1:size(qpoints, 1)-1
            stretch_energy = 10 * 0.5 * n * (lens[i] - ds)^2
            E += stretch_energy
            Estretch += stretch_energy
        end
        
        # Bending energy
        angles = calcanglecables(qpoints)
        for i in 1:size(qpoints, 1)-2
            clamped_angle = clamp(angles[i], -1.0, 1.0)
            bending_energy = kb * 0.5 * (2 * r)^2 * (1 - clamped_angle)
            E += bending_energy
            Ebending += bending_energy
        end
    end
    
    # Repulsion term between threads
    a = 1e-3
    σ = 2 * r
    ε = 1
    
    for i in 1:1
        for j in 2:2
            for p1 in 1:3:length(threads[i])
                for p2 in 1:3:length(threads[j])
                    d = norm(threads[i][p1:p1+2] - threads[j][p2:p2+2])
                    z = log(1 + exp((σ - d) / a))^2
                    repulsion_energy = ε * z
                    E += repulsion_energy
                    Ebars += repulsion_energy
                end
            end
        end
    end
    
    return E, Estretch, Ebending, Ebars
end

# Optimization function
function optimize_elastica_energy(p0, n, wR, wC, dsR, dsC, r, l0r, l0c, kb; tol=1e-3, max_iter=1000)
    """Optimize thread configuration to minimize energy"""
    p0 = Float64.(p0)
    
    function objective(p)
        return euler_elastica_energy(p, n, wR, wC, dsR, dsC, r, l0r, l0c, kb)[1]
    end
    
    result = optimize(objective, p0, GradientDescent(), 
                     Optim.Options(g_tol=1e-4, iterations=500, 
                                  store_trace=true, show_trace=true, show_warnings=true))
    
    grad = ForwardDiff.gradient(objective, result.minimizer)
    println("Gradient norm at convergence: ", norm(grad))
    
    p_opt = result.minimizer
    E_min = result.minimum
    
    return p_opt, E_min
end

# Main execution
# Parameters
r = 0.5  # Thread radius
n = 18   # Number of points per thread
kb = 1.0 # Bending stiffness
wR = 0.5 # Row frequency
wC = 0.5 # Column frequency
l0r = 1.2 # Row rest length
l0c = 3.0 # Column rest length
A = 1.0   # Amplitude
number_of_threads = 2

# Create output folder
folder = "results"
mkpath(folder)

println("Processing l0r = $l0r, l0c = $l0c")

# Initialize unit cell
(threads0, dsR, dsC) = initializeUnitCell(number_of_threads, A, wR, wC, n)
flat_threads0 = vcat(threads0...)

# Compute initial energy
E0, Es0, Ebending0, Ebars0 = euler_elastica_energy(flat_threads0, n, wR, wC, dsR, dsC, r, l0r, l0c, kb)
println("Initial Energy (E0): $E0")
println("Initial Stretch Energy: $Es0")
println("Initial Bending Energy: $Ebending0")
println("Initial Bars Energy: $Ebars0")

# Optimize threads
(threads, E) = optimize_elastica_energy(flat_threads0, n, wR, wC, dsR, dsC, r, l0r, l0c, kb)
println("Optimized Energy (E): $E")

_, Estretch, Ebending, Ebars = euler_elastica_energy(threads, n, wR, wC, dsR, dsC, r, l0r, l0c, kb)
println("Stretch Energy: $Estretch")
println("Bending Energy: $Ebending")
println("Bars Energy: $Ebars")

# Process optimized threads
threads111 = vcat(threads...)
threads11 = [collect(part) for part in Iterators.partition(threads111, 3 * (n - 2) + 2)]
threads1 = addBoundarytoThreats(threads11, wR, wC)

threads000 = vcat(threads0...)
threads00 = [collect(part) for part in Iterators.partition(threads000, 3 * (n - 2) + 2)]
threadsi0 = addBoundarytoThreats(threads00, wR, wC)

threadsf = threads1

# Save results
timestamp = Dates.format(now(), "yyyy-mm-dd")
filename = joinpath(folder, "threads_n_$(n)_l0r_$(l0r)_l0c_$(l0c)_kb_$(kb).txt")
writedlm(filename, threadsf)
println("Results saved to: $filename")
# Calculate final lengths and distances
fr(s) = sqrt(1 + A^2 * (pi * wR)^2 * cos(pi * wR * s)^2)
fc(s) = sqrt(1 + A^2 * (pi * wC)^2 * cos(pi * wC * s)^2)
lenir = (quadgk(fr, 0, 1 / wR)[1])
lenic = (quadgk(fc, 0, 1 / wC)[1])

nn = div(length(threadsf[1]), 3)
lenr = sum(calclencables(reshape(threadsf[1], (3, nn))'))
lenc = sum(calclencables(reshape(threadsf[2], (3, nn))'))

ax0 = -threadsi0[1][1] + threadsi0[1][end-2]
ay0 = -threadsi0[2][2] + threadsi0[2][end-1]
ax = -threadsf[1][1] + threadsf[1][end-2]
ay = -threadsf[2][2] + threadsf[2][end-1]

# Calculate minimal distance between threads
points1 = reshape(threadsf[1], (3, nn))'
points2 = reshape(threadsf[2], (3, nn))'

# Plot results
fig = Figure(resolution=(1200, 600))
ax1 = Axis3(fig[1, 1], title="Initial Threads", aspect=(1, 1, 1))
ax2 = Axis3(fig[1, 2], title="Optimized Threads", aspect=(1, 1, 1))

plot_unit_cell!(ax1, vcat(threads0...), n, wR, wC)
plot_unit_cell!(ax2, vcat(threads...), n, wR, wC)

display(fig)
