#pragma once

#include "core/op_type.h"
#include "core/tensor.h"

namespace infini
{
    using KernelAttrs = std::tuple<Device, OpType::underlying_t>;

    class GraphObj;
    class OperatorObj : public Object
    {
        friend class GraphObj;

    protected:
        OpType type; //算子类型 （Add, Clip, Transpose...）
        TensorVec inputs; //输入张量列表
        TensorVec outputs; //输出张量列表
        vector<WRef<OperatorObj>> predecessors; //前驱算子（生产我的输入的算子）
        vector<WRef<OperatorObj>> successors; //后继算子（消费我的输出的算子）

    public:
        OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs); //构造函数
        virtual optional<vector<Shape>> inferShape(const TensorVec &inputs) = 0; //推断输出形状
        virtual vector<DataType> inferDataType(const TensorVec &inputs) const; //推断输出数据类型
        /**
         * @brief Constructs outputs (if requried) and check whether the operator is
         * valid.
         *
         * @param graph If graph is not nullptr, outputs should be created in this
         * function.
         */
        bool checkValid(GraphObj *graph);

    public: // getter and setter 
        const TensorVec &getInputs() const { return inputs; } //获取输入张量列表
        const TensorVec &getOutputs() const { return outputs; } //获取输出张量列表
        Tensor getInputs(size_t i) const { return inputs.at(i); } //获取第i个输入张量
        Tensor getOutput() const //获取第一个输出张量
        {
            IT_ASSERT(outputs.size() == 1, "Unimplemented");
            return outputs[0];
        }
        Tensor getOutput(size_t i) const //获取第i个输出张量
        {
            IT_ASSERT(i < outputs.size(), "Index exceeded");
            return outputs.at(i);
        }
        OpVec getPredecessors() const { return wrefs_to_refs(predecessors); } //获取前驱算子列表
        OpVec getSuccessors() const { return wrefs_to_refs(successors); } //获取后继算子列表
        OpType getOpType() const { return type; } //获取算子类型
        // HACK: set correct data type
        DataType getDType() const { return getInputs(0)->getDType(); } //获取输入数据类型
        DataType getOutDType() const { return getOutput()->getDType(); } //获取输出数据类型
        virtual int numInputs() const = 0; //获取输入张量数量
        virtual int numOutputs() const = 0; //获取输出张量数量

        /**
         * @brief Clone this operator and replace its inputs and outputs.
         *
         * @param newInputs
         * @param newOutputs
         * @return Operator
         */
        virtual Operator clone(const TensorVec &newInputs,
                               const TensorVec &newOutputs) const = 0;

    protected:
        optional<vector<Shape>> inferShape();
        vector<DataType> inferDataType() const;

    private:
        void addPredecessors(const Operator &op) { predecessors.emplace_back(op); }
        void addSuccessors(const Operator &op) { successors.emplace_back(op); }
        void removePredecessors(const Operator &op);
        void removeSuccessors(const Operator &op);
        void replaceInput(Tensor t1, Tensor t2);
    };
// 定义宏，用于克隆算子
#define OP_CLONE(OpObj)                                                \
    virtual Operator clone(const TensorVec &newInputs,                 \
                           const TensorVec &newOutputs) const override \
    {                                                                  \
        auto op = infini::make_ref<OpObj>(*this);                      \
        op->inputs = newInputs;                                        \
        op->outputs = newOutputs;                                      \
        op->predecessors.clear();                                      \
        op->successors.clear();                                        \
        IT_ASSERT(op->checkValid(nullptr));                            \
        return op;                                                     \
    }

} // namespace infini
