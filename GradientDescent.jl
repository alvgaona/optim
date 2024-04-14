using ForwardDiff
using LinearAlgebra

function gradient_descent(f, x0; α=0.01, tol=1e-8, max_iters=1000)
  x = copy(x0)
  f_val = f(x)

  iteration = 0

  for i = 1:max_iters
    iteration = i
    ∇f = ForwardDiff.gradient(f, x)

    if norm(∇f) < tol
      break
    end

    x = x - α * ∇f
    f_val = f(x)
  end

  return x, f_val, iteration
end


function main()
  # 1-dimensional function
  f(x) = x[1]^2 - x[1]
  x0 = [1000]

  x_min, f_min, iterations = @time gradient_descent(f, x0, α=0.1, max_iters=1000)
  println("x* = $x_min, f(x*) = $f_min ($iterations=iterations)")

  # 2-dimensional function
  g(x) = x[1]^2 + x[2]^2 - 1
  x0 = [1.0, 1.0]

  x_min, g_min, iterations = @time gradient_descent(g, x0, α=0.1, max_iters=1000)
  println("x* = $x_min, f(x*) = $g_min ($iterations=iterations)")
end

main()

