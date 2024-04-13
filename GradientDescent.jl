using ForwardDiff

function gradient_descent(f, x0; α=0.01, tol=1e-8, max_iters=1000)
  x = copy(x0)
  f_val = f(x)
  
  iteration = 0

  for i = 1:max_iters
    iteration = i
    ∇f = ForwardDiff.gradient(f, x)

    x_new = x - α * ∇f
    f_new = f(x_new)

    error = abs(f_new - f_val)

    if error < tol
      return x_new, f_new, iteration
    end

    x = x_new
    f_val = f_new
  end

  return x, f_val, iteration
end

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
