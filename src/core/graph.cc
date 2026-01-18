#include "core/graph.h"
#include "core/blob.h"
#include "operators/transpose.h"
#include "operators/matmul.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include <set>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        
        // ========== 策略说明 ==========
        // 使用"标记-清除"策略：
        // 1. 先遍历图，标记需要删除的算子和张量（不立即删除，避免迭代器失效）
        // 2. 最后统一清理引用关系并删除
        // ==============================
        
        std::set<Operator> opsToRemove;      // 标记要删除的算子
        std::set<Tensor> tensorsToRemove;    // 标记要删除的张量
        
        // ==================== 优化1：去除相邻的可抵消的 Transpose ====================
        // 原理：如果两个相邻的 Transpose 具有相同的 permute，例如：
        //   Transpose({0,1,3,2}) → Transpose({0,1,3,2})
        // 执行两次相同的转置操作等于恢复原状，可以直接消除这两个算子
        // 
        // 图变化示意：
        //   优化前: i1 → [Transpose7] → t1 → [Transpose8] → t2 → [后续算子]
        //   优化后: i1 ────────────────────────────────────────→ [后续算子]
        // ===========================================================================
        for (auto &op : ops)
        {
            // 跳过已标记删除的算子
            if (opsToRemove.count(op)) continue;
            // 只处理 Transpose 类型的算子
            if (op->getOpType() != OpType::Transpose) continue;
            
            // 将算子转换为 TransposeObj 类型，获取更多信息
            auto transposeOp = as<TransposeObj>(op);
            // 获取当前 Transpose 的输入张量
            Tensor inputTensor = transposeOp->getInputs(0);
            // 获取产生这个输入张量的算子（即前一个算子）
            Operator sourceOp = inputTensor->getSource();
            
            // 检查前一个算子是否存在且未被标记删除
            if (!sourceOp || opsToRemove.count(sourceOp)) continue;
            // 检查前一个算子是否也是 Transpose
            if (sourceOp->getOpType() != OpType::Transpose) continue;
            
            // 获取前一个 Transpose 算子
            auto prevTransposeOp = as<TransposeObj>(sourceOp);
            // 获取两个 Transpose 的 permute 向量
            auto perm1 = transposeOp->getPermute();      // 当前 Transpose 的 permute
            auto perm2 = prevTransposeOp->getPermute();  // 前一个 Transpose 的 permute
            
            // 如果两个 permute 相同，说明可以抵消
            // 例如：perm1 = perm2 = {0,1,3,2}，两次转置后恢复原状
            if (perm1 == perm2)
            {
                // 获取相关的张量：
                // originalTensor: 第一个 Transpose 的输入（最原始的张量）
                // middleTensor: 两个 Transpose 之间的中间张量
                // outputTensor: 第二个 Transpose 的输出
                auto originalTensor = prevTransposeOp->getInputs(0);
                auto middleTensor = inputTensor;
                auto outputTensor = transposeOp->getOutput();
                
                // 重新连接：让使用 outputTensor 的后续算子改为使用 originalTensor
                // 这样后续算子就直接使用原始输入，跳过了两个 Transpose
                for (auto &succ : outputTensor->getTargets())
                {
                    succ->replaceInput(outputTensor, originalTensor);
                }
                
                // 标记要删除的张量和算子
                tensorsToRemove.insert(middleTensor);    // 删除中间张量
                tensorsToRemove.insert(outputTensor);    // 删除输出张量
                opsToRemove.insert(prevTransposeOp);     // 删除第一个 Transpose
                opsToRemove.insert(transposeOp);         // 删除第二个 Transpose
            }
        }
        
        // ==================== 优化2：将 Transpose 融合到 Matmul 中 ====================
        // 原理：Matmul 算子有 transA 和 transB 属性，如果输入的 Transpose 只是
        //      交换最后两个维度，可以直接设置 trans 属性，省去 Transpose 算子
        // 
        // 条件：Transpose 的 permute 必须是只交换最后两维
        //   例如：4维张量的 permute = {0, 1, 3, 2}（前两维不变，后两维交换）
        // 
        // 图变化示意：
        //   优化前: i2 → [Transpose9] → t3 → [Matmul(A, B)]
        //   优化后: i2 ───────────────────→ [Matmul(A, B^T)]  (设置 transB=true)
        // ===========================================================================
        for (auto &op : ops)
        {
            // 跳过已标记删除的算子
            if (opsToRemove.count(op)) continue;
            // 只处理 Transpose 类型的算子
            if (op->getOpType() != OpType::Transpose) continue;
            
            auto transposeOp = as<TransposeObj>(op);
            auto outputTensor = transposeOp->getOutput();  // Transpose 的输出张量
            auto perm = transposeOp->getPermute();         // 获取 permute 向量
            int rank = perm.size();                        // 张量的维度数
            
            // 至少需要2维才能交换最后两维
            if (rank < 2) continue;
            
            // 检查 permute 是否只交换最后两维
            // 条件：perm[rank-1] == rank-2 且 perm[rank-2] == rank-1
            //       其他维度保持不变：perm[i] == i (对于 i < rank-2)
            // 例如：{0, 1, 3, 2} 满足条件，{0, 2, 1, 3} 不满足
            bool isLastTwoDimSwap = (perm[rank - 1] == rank - 2) && (perm[rank - 2] == rank - 1);
            for (int i = 0; i < rank - 2; ++i)
            {
                if (perm[i] != i) { isLastTwoDimSwap = false; break; }
            }
            
            // 如果不是只交换最后两维，跳过
            if (!isLastTwoDimSwap) continue;
            
            // 查找使用这个 Transpose 输出的下游算子
            for (auto &targetOp : outputTensor->getTargets())
            {
                // 只处理 MatMul 类型的下游算子
                if (targetOp->getOpType() != OpType::MatMul) continue;
                
                auto matmulOp = as<MatmulObj>(targetOp);
                auto transposeInput = transposeOp->getInputs(0);  // Transpose 的原始输入
                
                // 判断 Transpose 的输出是 Matmul 的 A 输入还是 B 输入
                if (matmulOp->getInputs(0) == outputTensor)
                {
                    // 是 A 输入：设置 transA 属性（取反，因为原来可能已经有值）
                    matmulOp->setTransA(!matmulOp->getTransA());
                    // 替换输入：Matmul 直接使用 Transpose 的原始输入
                    matmulOp->replaceInput(outputTensor, transposeInput);
                }
                else if (matmulOp->getInputs(1) == outputTensor)
                {
                    // 是 B 输入：设置 transB 属性
                    matmulOp->setTransB(!matmulOp->getTransB());
                    matmulOp->replaceInput(outputTensor, transposeInput);
                }
                
                // 标记要删除的张量和算子
                tensorsToRemove.insert(outputTensor);
                opsToRemove.insert(transposeOp);
                break;  // 已处理，退出内层循环
            }
        }
        
        // ==================== 清理引用关系 ==================== 
        // 重要！删除算子前必须清理所有指向它的引用，否则会导致悬空指针崩溃
        // 
        // 图中的引用关系：
        // - 张量的 targets 列表记录了使用该张量的算子
        // - 张量的 source 记录了产生该张量的算子
        // - 算子的 predecessors 记录了前驱算子
        // - 算子的 successors 记录了后继算子
        // 
        // 如果不清理这些引用，后续打印图或遍历时会访问已删除的对象！
        // =======================================================
        for (auto &opToRemove : opsToRemove)
        {
            // 1. 从输入张量的 targets 列表中移除此算子
            //    （告诉输入张量：我不再使用你了）
            for (auto &input : opToRemove->getInputs())
            {
                if (input) input->removeTarget(opToRemove);
            }
            
            // 2. 从输出张量的 source 中移除此算子
            //    （告诉输出张量：我不再是你的生产者了）
            for (auto &output : opToRemove->getOutputs())
            {
                if (output) output->setSource(nullptr);
            }
            
            // 3. 从前驱算子的 successors 列表中移除此算子
            //    （告诉前驱：我不再是你的后继了）
            for (auto &pred : opToRemove->getPredecessors())
            {
                pred->removeSuccessors(opToRemove);
            }
            
            // 4. 从后继算子的 predecessors 列表中移除此算子
            //    （告诉后继：我不再是你的前驱了）
            for (auto &succ : opToRemove->getSuccessors())
            {
                succ->removePredecessors(opToRemove);
            }
        }
        
        // ==================== 批量删除 ====================
        // 从图的 tensors 和 ops 列表中移除被标记的对象
        for (auto &t : tensorsToRemove) removeTensor(t);
        for (auto &o : opsToRemove) removeOperator(o);
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // ==================== 内存分配流程 ====================
        // 1. 首先进行拓扑排序，确保算子按正确的执行顺序排列
        // 2. 为每个张量分配内存空间（通过 allocator）
        // 3. 将实际的内存指针绑定到张量
        // 
        // 内存管理架构：
        //   allocator 管理一块连续的内存池
        //   每个张量获得一个偏移量（offset），表示它在内存池中的位置
        //   实际地址 = 内存池基地址 + 偏移量
        // ======================================================
        
        // 拓扑排序，确保算子按依赖关系排序
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        
        // ========== 第一步：为每个张量分配空间 ==========
        // allocator.alloc(size) 会返回一个偏移量（offset）
        // 这个偏移量是相对于内存池起始地址的位置
        // 需要先记录所有偏移量，因为 getPtr() 必须在所有 alloc 之后调用
        std::vector<size_t> offsets;
        for (auto &tensor : tensors)
        {
            // 获取张量需要的字节数（元素数量 × 数据类型大小）
            size_t size = tensor->getBytes();
            // 调用 allocator 分配内存，返回偏移量
            // 注意：alloc 只是"登记"需要的空间，记录偏移量，不实际分配内存
            size_t offset = allocator.alloc(size);
            offsets.push_back(offset);
        }
        
        // ========== 第二步：获取实际的内存基地址 ==========
        // getPtr() 会根据之前 alloc 记录的 peak 值，向 runtime 申请实际的内存
        // 返回的 basePtr 是整个内存池的起始地址
        void *basePtr = allocator.getPtr();
        
        // ========== 第三步：为每个张量设置数据指针 ==========
        for (size_t i = 0; i < tensors.size(); ++i)
        {
            // 计算实际地址：内存池基地址 + 该张量的偏移量
            // 注意：必须将 basePtr 转为 char* 才能进行字节级别的指针运算
            void *ptr = static_cast<char*>(basePtr) + offsets[i];
            
            // 创建 BlobObj 对象并绑定到张量
            // BlobObj 是一个包装类，持有指针和 runtime 信息
            tensors[i]->setDataBlob(make_ref<BlobObj>(runtime, ptr));
        }
        
        // 打印内存分配信息（用于调试）
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini