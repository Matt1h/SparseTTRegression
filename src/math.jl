function LinearAlgebra.pinv(A::MPS)  # TODO: only works if RowDim/ColDim is the last siteind
    sites = siteinds(A)
    single_ind, single_ind_loc, _ = find_matrix_size(sites)

    W = deepcopy(A)

    # new tag
    if hastags(single_ind, "RowDim")
        W[single_ind_loc] = replaceind(W[single_ind_loc], single_ind, replacetags(single_ind, "RowDim", "ColDim"))
    elseif hastags(single_ind, "ColDim")
        W[single_ind_loc] = replaceind(W[single_ind_loc], single_ind, replacetags(single_ind, "ColDim", "RowDim"))
    end

    idx = single_ind_loc - 1

    orthogonalize!(W, idx)

    if order(A[idx]) == 2
        u, s, v = svd(W[idx], inds(W[idx])[1])
    elseif order(W[idx]) == 3
        u, s, v = svd(W[idx], inds(W[idx])[1:2])
    end

    W[idx] = u

    W[idx+1] = v*W[idx+1]
    W[idx+1] = (1 ./ s)*W[idx+1]
    return W
end





