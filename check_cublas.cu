#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    cublasHandle_t handle;
    cublasStatus_t status;

    // Inicializar CUBLAS
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS no está instalado o no funciona correctamente.\n");
        return -1;
    }

    // Ejemplo sencillo: Escalar un vector
    const int n = 10;
    float alpha = 2.0f;
    float x[n] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    float *d_x;

    // Asignar memoria en la GPU
    cudaMalloc((void **)&d_x, n * sizeof(float));

    // Copiar datos al dispositivo
    cublasSetVector(n, sizeof(float), x, 1, d_x, 1);

    // Escalar el vector x por alpha y almacenar el resultado en x
    cublasSscal(handle, n, &alpha, d_x, 1);

    // Copiar los resultados de vuelta a la memoria del host
    cublasGetVector(n, sizeof(float), d_x, 1, x, 1);

    // Limpiar
    cudaFree(d_x);
    cublasDestroy(handle);

    printf("CUBLAS está instalado y funcionando correctamente.\n");
    return 0;
}
