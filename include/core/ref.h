#pragma once
#include "core/common.h"
#include <functional>
#include <memory>
#include <type_traits>

namespace infini {

template <typename T> using Ref = std::shared_ptr<T>; //强引用
template <typename T> using WRef = std::weak_ptr<T>; //弱引用
//类型检测
template <typename T> struct is_ref : std::false_type {};
template <typename T> struct is_ref<Ref<T>> : std::true_type {};
template <typename T> struct is_ref<WRef<T>> : std::true_type {};

//创建对象并返回智能指针
template <typename T, typename... Params> Ref<T> make_ref(Params &&...params) {
    static_assert(is_ref<T>::value == false, "Ref should not be nested");
    return std::make_shared<T>(std::forward<Params>(params)...);
}
//类型转换
template <class T, class U,
          typename std::enable_if_t<std::is_base_of_v<U, T>> * = nullptr>
Ref<T> as(const Ref<U> &ref) {
    return std::dynamic_pointer_cast<T>(ref);
}

template <typename T>
std::vector<WRef<T>> refs_to_wrefs(const std::vector<Ref<T>> &refs) {
    std::vector<WRef<T>> wrefs;
    for (const auto &ref : refs)
        wrefs.emplace_back(ref);
    return wrefs;
}

template <typename T>
std::vector<Ref<T>> wrefs_to_refs(const std::vector<WRef<T>> &wrefs) {
    std::vector<Ref<T>> refs;
    for (const auto &wref : wrefs)
        refs.emplace_back(wref);
    return refs;
}

} // namespace infini
