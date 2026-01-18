#include "operators/matmul.h"
#include "utils/operator_utils.h"

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
        
        int rankA = shape_A.size();
        int rankB = shape_B.size();
        
        // 获取 A 和 B 的最后两维（矩阵维度）
        int m = transA ? shape_A[rankA - 1] : shape_A[rankA - 2];
        int kA = transA ? shape_A[rankA - 2] : shape_A[rankA - 1];
        int kB = transB ? shape_B[rankB - 1] : shape_B[rankB - 2];
        int n = transB ? shape_B[rankB - 2] : shape_B[rankB - 1];
        
        // 获取 batch 维度
        Shape batchA(shape_A.begin(), shape_A.end() - 2);
        Shape batchB(shape_B.begin(), shape_B.end() - 2);
        
        // 对 batch 维度进行广播
        Shape batchC;
        if (batchA.empty() && batchB.empty()) {
            // 都是 2D 矩阵，无 batch
        } else if (batchA.empty()) {
            batchC = batchB;
        } else if (batchB.empty()) {
            batchC = batchA;
        } else {
            batchC = infer_broadcast(batchA, batchB);
        }
        
        // 构建输出 shape
        Shape shape_C = batchC;
        shape_C.push_back(m);
        shape_C.push_back(n);
        
        return {{shape_C}};
    }

} // namespace infini