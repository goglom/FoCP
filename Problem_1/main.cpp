#include <iostream>
#include <limits>
#include <concepts>
#include <iomanip>

using namespace std;

template<class T>
std::string TypeName() {
  return  typeid(T).name(); 
}

template<std::floating_point T>
pair<T, size_t>  CalcULP_MantissaSize() {
  T one(1), ulp(1);
  size_t mantissa_bits(1);

  while (one + ulp / 2 != one) {
    ulp /= 2;
    ++mantissa_bits;
  }
  
  return {ulp, mantissa_bits};
}

template<std::floating_point T>
int CalcMaxExponent() {
  T val(1);
  int exponentPow = 0;

  while (val * 2 != val) {
    val *= 2;
    ++exponentPow;
  }
  return exponentPow;
}

template<std::floating_point T>
int CalcMinExponent() {
  T val(1);
  T zero(0);
  int exponentPow = 0;

  while (val / 2 != zero) {
    val /= 2;
    --exponentPow;
  }
  return exponentPow;
}


template<std::floating_point T>
void PrintTaskResults() {
  auto [ulp, mantissa_size] = CalcULP_MantissaSize<T>();
  cout << setprecision(mantissa_size);

  cout << "<" + TypeName<T>() + "> type inforamtion:" << endl;
  cout << " - Size:                   " << sizeof(T) * 8 << " bits" << endl;
  cout << " - My ulp:                 " << ulp << endl;
  cout << " - STD ulp:                " << numeric_limits<T>::epsilon() << endl;
  cout << " - My mantissa digits:     " << mantissa_size << endl;
  cout << " - STD mantissa digits:    " << numeric_limits<T>::digits << endl;
  cout << " - My exponent max power:  " << CalcMaxExponent<T>() << endl;
  cout << " - STD exponent max power: " << numeric_limits<T>::max_exponent << endl;
  cout << " - My exponent min power:  " << CalcMinExponent<T>() << endl;
  //cout << " - STD exponent min power: " << numeric_limits<T>::min_exponent - numeric_limits<T>::digits << endl;
  cout << "            1 = " << T(1) << endl;
  cout << "      1 + e/2 = " << T(1) + ulp/2 << endl;
  cout << "        1 + e = " << T(1) + ulp << endl;
  cout << "  1 + e + e/2 = " << (T(1) + ulp) + ulp / 2 << endl;

  cout << endl;
}

int main() {
  cout << scientific;
  PrintTaskResults<float>();
  PrintTaskResults<double>();

  return 0;
}
