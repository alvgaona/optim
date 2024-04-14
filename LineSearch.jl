using ForwardDiff: gradient
using LinearAlgebra

function backtracking(f, x, t₀=1; Δx, β=0.5, α=0.25)
  """
  Performs backtracking line search to find a suitable step length.

  Args:
    f (Function): the objective function
    x (Vector): the current point
    t₀ (Float64): initial step length
    Δx (Vector): the search direction
    β (Float64): backtracking factor. β ∈ (0,1)
    α (Float64): sufficient decrease condition parameter. α ∈ (0, 0.5)
  """

  t = t₀
  fₓ = f(x)

  while f(x + t * Δx) > fₓ + α * t * dot(gradient(f, x), Δx)
    t *= β
  end

  return t
end


f(x) = x[1]^2 + x[2]^2 - 1

x = [2.5 2.5]

t = backtracking(f, x, Δx=gradient(f, x), β=0.1, α=0.1)

println("Step length is $t")
