function ttmtest()
    g = gpuDevice(1);

    disp 'Read A from testa.tns';
    fp = fopen('testa.tns', 'r');
    X = spTensor.load(1, fp);
    fclose(fp);

    disp 'Read B from testb.tns';
    fp = fopen('testb.tns', 'r');
    U = spTensor.load(1, fp);
    U = U.toMatrix();
    fclose(fp);

    Y = X.timesMatrix(U, 2);
    Y = spTensor.fromSspTensor(Y, 1e-6);
    disp 'Write A*B to testy.tns (CPU kernel)';
    fp = fopen('testy.tns', 'w');
    Y.dump(1, fp);
    fclose(fp);

    Y = X.timesMatrixCuda(U, 2);
    Y.values = gather(Y.values);
    Y = spTensor.fromSspTensor(Y, 1e-6);
    disp 'Write A*B to testcuda.tns (CUDA kernel)';
    fp = fopen('testcuda.tns', 'w');
    Y.dump(1, fp);
    fclose(fp);
end

