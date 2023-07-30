#ifndef COMMON_H_
#define COMMON_H_

#define INSTANTIATE_CLASS(classname) \
template class classname<double>; \
template class classname<float>;


#endif