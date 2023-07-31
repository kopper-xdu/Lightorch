#pragma once

#define INSTANTIATE_CLASS(classname) \
template class classname<double>; \
template class classname<float>;