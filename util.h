#ifndef NN_UTIL_H_MGOMEZ4
#define NN_UTIL_H_MGOMEZ4
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <assert.h>

using namespace std;
// g++ -O -std=c++11 -Wall -Wextra -pedantic  -o nn nn.cpp
// transpose matrix
void transpose_2d(vector<vector<float>> &vect)
{
    // assumes non-empty
    // temp vector to tranpose ofrom
    vector<vector<float>> temp(vect[0].size(), vector<float>());
    
    for (int i = 0; i < vect.size(); i++){
    #pragma omp parallel for
        for (int j = 0; j < vect[0].size(); j++)
            temp[j].push_back(vect[i][j]);
    }
    vect = temp;
}


/* READING DATA */
int reverseInt(int i)
{
  unsigned char c1, c2, c3, c4;

  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;

  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

vector<float> readLabels(string filename)
{
  ifstream file(filename, ios::binary);
  int magic_num = 0;
  int num_items = 0;
  file.read((char *)&magic_num, sizeof(magic_num));
  file.read((char *)&num_items, sizeof(num_items));
  magic_num = reverseInt(magic_num);
  num_items = reverseInt(num_items);
  vector<float> labels;

  for (int i = 0; i < num_items; i++)
  {
    unsigned char x;
    unsigned int label;
    file.read((char *)&x, 1);
    label = static_cast<unsigned int>(x);
    labels.push_back(static_cast<float>(label));
  }
  file.close();
  return labels;
}

vector<vector<float>> readImages(string filename)
{
  ifstream file(filename, ios::binary);
  int magic_num = 0;
  int num_items = 0;
  int num_cols = 0;
  int num_rows = 0;
  file.read((char *)&magic_num, sizeof(magic_num));
  file.read((char *)&num_items, sizeof(num_items));
  file.read((char *)&num_rows, sizeof(num_rows));
  file.read((char *)&num_cols, sizeof(num_cols));
  magic_num = reverseInt(magic_num);
  num_items = reverseInt(num_items);
  num_rows = reverseInt(num_rows);
  num_cols = reverseInt(num_cols);
  vector<vector<float>> images;

  for (int i = 0; i < num_items; i++)
  {
    vector<float> image;
    for (int j = 0; j < num_cols * num_rows; j++)
    {
      unsigned char x;
      unsigned int pixel;
      file.read((char *)&x, 1);
      pixel = static_cast<unsigned int>(x);
      image.push_back((static_cast<float>(pixel))/255.0);// @HERE 255
    }
    images.push_back(image);
  }
  file.close();
  return images;
}

void resize_2d(vector<vector<float>> &v, int N, int M){
    v.resize(N);
    for (int i = 0; i < N; ++i)
        v[i].resize(M);
}
void fill_row(std::vector<float> &row)
{
    generate(row.begin(), row.end(), [](){ return d(gen);});
}

void fill_matrix(std::vector<std::vector<float>> &v)
{
    for_each(v.begin(), v.end(), fill_row);
}

void encode_soln(vector<float> solns, vector<vector<float>>&v, int samples){

    resize_2d(v,samples, 10); // v is samples * 10
    assert(v.size() == samples && v[0].size() == 10);
    assert(solns.size()==samples);
    for(int i = 0; i < solns.size(); i++){
        int index = solns[i];
        v[i][index] = 1;
        for(int j =0; j < 10; j++){
            if(j!= index)  v[i][j] = 0;
        }
    }
    transpose_2d(v);
    assert(v.size()==10);
}


#endif