#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto shape_A = inputs[0]->getDims();
        auto shape_B = inputs[1]->getDims();
        auto shape_C = shape_A;
        bool transA = getTransA();
        bool transB = getTransB();
        for(size_t i = 0; i < shape_A.size() - 2; i++)
        {
            shape_C[i] = std::max(shape_A[i], shape_B[i]);
        }

        shape_C[shape_A.size() - 2] = transA ? shape_A[shape_A.size() - 1] : shape_A[shape_A.size() - 2];
        shape_C[shape_A.size() - 1] = transB ? shape_B[shape_B.size() - 2] : shape_B[shape_B.size() - 1];
        return {{shape_C}};
    }

} // namespace infini