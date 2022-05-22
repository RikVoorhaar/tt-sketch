function [x, ranks] = TTGMRES(A, bo, tol, maxit, prec, tols)
% TTGMRES  Compute the solution of a linear system in Kronecker form using the TT-GMRES algorithm.


k = 0;
normb = TTnorm(bo);
V = cell(maxit, 1);

b = bo;

r = b;
normres = TTnorm(r);
V{1} = TTscale(r, 1/normres);


H = zeros(maxit + 1, maxit);
res = zeros(maxit + 1, 1);
res(1) = normres;

rK = zeros(maxit + 1, 1);
rK(1) = normres;

ranks = cell(maxit + 2, 1);
ranks{1} = TTranks(V{1});


while(k < maxit && normres > tol * normb)

    k = k + 1;
    w = prec(V{k});
    w = A(w);

    delta = 0.01*tol/(normres * normb);
    w = TTrounding_Sum_Randomize_then_Orthogonalize(w, ones(length(w)), delta);

    h = zeros(k + 1, 1);
    for j = 1 : k
        h(j) = innerProdTT(V{j}, w);
    end

    V{k+1} = w;
    w = TTrounding_Sum_Randomize_then_Orthogonalize(V(1:k+1), [-h(1:k);1], delta);

    h(k + 1) = TTnorm(w);
    if(h(k + 1) < 2e-16)
        keyboard
    end
    V{k + 1} = TTscale(w, 1/h(k + 1));
    ranks{k + 1} = TTranks(V{k+1});
    
    H(1 : k + 1, k) = h;
    [Q, R] = qr(H(1 : k + 1, 1 : k));
    t = Q' * rK(1 : k + 1);
    res(k + 1) = abs(t(k + 1));
    normres = res(k + 1);
    
    t = Q' * rK(1 : k + 1);
end


y = R(1 : k, 1 : k)\t(1 : k);
x = TTrounding_Sum_Randomize_then_Orthogonalize(V(1:k), y(1:k), tols);
x = prec(x);
x = TTrounding(x, tols);
ranks{k + 2} = TTranks(x);
ranks = ranks(1 : k + 2);