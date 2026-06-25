function [d, W, transform] = procrustNew(X, Y)

    [n, m] = size(X);
    [ny, my] = size(Y);

    muX = mean(X,1);
    muY = mean(Y,1);

    X0 = X - repmat(muX, n, 1);
    Y0 = Y - repmat(muY, n, 1);

    ssX = sum(X0.^2, 1);
    ssY = sum(Y0.^2, 1);
    normX = sqrt(sum(ssX));
    normY = sqrt(sum(ssY));

    normX = max(normX, eps);
    normY = max(normY, eps);

    X0 = X0 / normX;
    Y0 = Y0 / normY;

    A = X0' * Y0;

    A(isnan(A)) = 0;
    A(isinf(A)) = 0;

    [U, S, V] = svd(A);

    T = V * U';

    b = trace(S) * normX / normY;

    W = b * T;

    c = muX - b * muY * T;

    Z = b * Y * T + repmat(c, n, 1);

    sum_ssX = sum(ssX);
    sum_ssX = max(sum_ssX, eps);

    d = sum(sum((X - Z).^2)) / sum_ssX;

    transform.T = T;
    transform.b = b;
    transform.c = c;

end