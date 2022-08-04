function init_H_mals(U::MPS, A::MPS)
	order = length(U)
	H = Array{ITensor}(undef, order-1)
	H[order-1] = A[order+1]*A[order]
	for q = order-1:-1:2
		H[q-1] = U[q+1]*A[q]*H[q]  # update H
	end
	return H
end


function Aq_full(Gq::ITensor, Hq::ITensor)
	Aq = Gq*Hq
	Aq_inds = inds(Aq)

	ColInd = filter(x -> hastags(x, "ColDim"), Aq_inds)
	V_inds = filter(x -> !hastags(x, "ColDim"), Aq_inds)

	col_loc = findall(x -> hastags(x, "ColDim"), Aq_inds)
	non_col_loc = findall(x -> !hastags(x, "ColDim"), Aq_inds)

	Aq_array = permutedims(Array(Aq, Aq_inds), [col_loc; non_col_loc])
	return reshape(Aq_array, ITensors.dim(ColInd), ITensors.dim(V_inds)), V_inds
end


function right_core_move_mals(U::MPS, V::ITensor, q::Integer, tol::Number)
	if order(V) == 3 && q == 1
		u, s, v = svd(V, inds(V)[1], cutoff=tol)  # TODO: maxdim instead of cutoff also possible
	elseif order(V) == 4 || (order(V) == 3 && q > 1)
		u, s, v = svd(V, inds(V)[1:2], cutoff=tol)
	end
	U[q] = u
	U[q+1] = s*v
	return U
end


function left_core_move_mals(U::MPS, V::ITensor, q::Integer, tol::Number)
	if order(V) == 3 && q == 1
		u, s, v = svd(V, inds(V)[1], cutoff=tol)  # TODO: maxdim instead of cutoff also possible
	elseif order(V) == 4 || (order(V) == 3 && q > 1)
		u, s, v = svd(V, inds(V)[1:2], cutoff=tol)
	end
	U[q+1] = v
	U[q] = u*s
	return U
end


function cost_function(Gq::ITensor, Hq::ITensor, V::ITensor, U::MPS, b::AbstractVector, λ::Number)
	K = Gq*Hq*V
	return sqrt(sum((K - ITensor(b, inds(K))).^2)) + λ*sum(abs.(Matrix(U)))
end


function mals_lasso(tt_opt::MPS, A::MPS, b::AbstractVector, λ::AbstractVector; verbose=false::Bool,
	svd_tol=1e-14::Number, lasso_tol=1e-7, sweep_count=10)

    order = length(tt_opt)
	orthogonalize!(tt_opt, 1)

	# Initialize 
	G = repeat([ITensor()], order-1)
	G[1] = A[1]

	#Initialize H
	H = init_H_mals(tt_opt, A)
	
	nsweeps = 0  # sweeps counter

	trace_forward = []
	trace_backward = []

	while nsweeps < sweep_count
		nsweeps+=1
		# First half sweep
		for q = 1:(order-1)
			verbose ? println("Forward sweep: core optimization $q out of ", order-1,) : nothing

			# Lasso optimization of core q
			Aq, V_inds = Aq_full(G[q], H[q])  # TODO: cotrol ranksize
			f = fit(LassoPath, Aq, b, λ=λ, maxncoef=ITensors.dim(V_inds), cd_tol=lasso_tol, standardize = false, intercept = false)

			# reshape and insert updated core
			V_m = Matrix(f.coefs)[:, length(λ)]
			V = ITensor(V_m, V_inds)
			tt_opt = right_core_move_mals(tt_opt, V, q, svd_tol)
			cost = cost_function(G[q], H[q], V, tt_opt, b, λ[end])
			push!(trace_forward, cost)
			verbose ? print("cost function = ", cost, "\n") : nothing

			#update G
			q < order-1 ? G[q+1] = G[q]*tt_opt[q]*A[q+1] : nothing
		end

		if nsweeps == sweep_count
			return tt_opt
		else
			nsweeps+=1
			# Second half sweep
			for q = order-1:(-1):1
				verbose ? println("Bakward sweep: core optimization $q out of ", order-1,) : nothing

				# Lasso optimization of core q
				Aq, V_inds = Aq_full(G[q], H[q])  # TODO: cotrol ranksize
				f = fit(LassoPath, Aq, b, λ=λ, maxncoef=ITensors.dim(V_inds), cd_tol=lasso_tol, standardize = false, intercept = false)

				# reshape and insert updated core
				V_m = Matrix(f.coefs)[:, length(λ)]
				V = ITensor(V_m, V_inds)
				tt_opt = left_core_move_mals(tt_opt, V, q, svd_tol)
				cost = cost_function(G[q], H[q], V, tt_opt, b, λ[end])
				push!(trace_backward, cost)
				verbose ? print("cost function = ", cost, "\n") : nothing
				
				# update H
				q > 1 ? H[q-1] = tt_opt[q+1]*A[q]*H[q] : nothing
			end
		end
	end
	return tt_opt, trace_forward, trace_backward
end