function ttmtest()
    g = gpuDevice(1);

    fp = fopen('testa.tns', 'r');
    X = spTensor.load(1, fp);
    fclose(fp);

    fp = fopen('testb.tns', 'r');
    U = spTensor.load(1, fp);
    U = U.toMatrix();
    fclose(fp);

    Y = X.timesMatrix(U, 2);
    Y = spTensor.fromSspTensor(Y, 1e-6);
    fp = fopen('testy.tns', 'w');
    Y.dump(1, fp);
    fclose(fp);

    Y = X.timesMatrixCuda(U, 2);
    Y.values = gather(Y.values);
    Y = spTensor.fromSspTensor(Y, 1e-6);
    fp = fopen('testcuda.tns', 'w');
    Y.dump(1, fp);
    fclose(fp);
end

