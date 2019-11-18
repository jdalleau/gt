
#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <vector>
#include <ctime>
#include <string>
#include <queue>
#include <fstream>
#include <sstream>
#include <iterator>
#include <cstdio>
#include "./Eigen/Eigen"
#include <chrono>
#include <omp.h>
#include <fstream>
#include <unordered_set>

using Eigen::MatrixXd;

using namespace std;
using std::milli;
using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::sort;


int performSplit(const vector<vector<int>> &dataset, vector<int> &attributesIndices,
                 const vector<int> &attributes, vector<int> &left, vector<int> &right, const vector<int> &nodeIndices)
{
  
  std::random_device rd;
  std::mt19937_64 mt(rd());
  std::uniform_int_distribution<int> dist (0,attributesIndices.size()-1);
  
  //Sample a random index column
  int randIndex = dist(mt);
  
  //Get the corresponding index attribute
  int attributeIndex = attributesIndices[randIndex];
  
  *attributesIndices.erase(attributesIndices.begin() + randIndex);
  int attribute = attributes[attributeIndex];
  std::unordered_set<float> set;
  

  //Fill the indices
  for (int i = 0; i < nodeIndices.size(); i++)
  {   
      int value = dataset[nodeIndices[i]][attribute];
      set.insert(value);
      if (value == 1)
      { 
        left.push_back(i);
      }
      else
      {
        right.push_back(i);
      }
  }
  //If there is only one value:
  if (set.size() <= 1)
  {
    return 1;
  }
  else 
  {
  return 0;
  }
}

struct Node
{
  vector<int> indices;
  vector<int> instances;
};

MatrixXd build_randomized_tree_and_get_sim(const vector<vector<int>> &data,
                                           const int &nmin, int nTrees)
{
  int nrows = data.size();
  cout << nrows << endl;
  vector<int> firstVector=data[0];
  int ncols=firstVector.size();
  cout << ncols << endl;
  // MatrixXd matrix2 = MatrixXd::Zero(nrows, nrows);
  vector<int> nodeIndices;
  for (int i=0; i < nrows; i++) {
    nodeIndices.push_back(i);
  }
  MatrixXd matrix = MatrixXd::Zero(nrows, nrows);

  #pragma omp parallel for
  for (int loop = 0; loop < nTrees; loop++)
  {

    list<Node> nodes;

    vector<int> attributes, attributes_indices, instanceList;
    for (int i = 0; i < ncols; i++)
    {
      attributes_indices.push_back(i);
      attributes.push_back(i);
    }

    for (int i=0; i < nrows; i++) {
      instanceList.push_back(i);
    }


    vector<int> left_indices, right_indices, left_instances, right_instances;
    performSplit(data, attributes_indices, attributes, left_indices, right_indices, nodeIndices);

    for (int i : left_indices) {
      left_instances.push_back(instanceList[i]);
    }
    for (int i : right_indices) {
      right_instances.push_back(instanceList[i]);
    }

    if (left_indices.size() < nmin and left_instances.size() != 0)
    {
      for (int instance1 : left_instances)
      {
        for (int instance2 : left_instances)
        {
          matrix(instance1, instance2) += left_instances.size();
        }
      }
    }

    else
    {
      Node currentNode = {left_indices, left_instances};
      nodes.push_back(currentNode);
    }

    if (right_indices.size() < nmin and right_instances.size() != 0)
    {
      for (int instance1 : right_instances)
      {

        for (int instance2 : right_instances)
        {
          matrix(instance1, instance2) += right_instances.size();
        }
      }
    }
    else
    {
      Node currentNode = {right_indices, right_instances};
      nodes.push_back(currentNode);
    }
    for (int instance1 : left_instances) {
        for (int instance2 : right_instances) {
            matrix(instance1,instance2) += (left_instances.size() + right_instances.size());
            matrix(instance2,instance1) += (left_instances.size() + right_instances.size());
        }
    }
    // Root node successfully has two children. Now, we iterate over these children.
    while (!nodes.empty())
    {
      if (attributes_indices.size() < 1)
      {
        for (Node node : nodes)
        {
          vector<int> instances = node.instances;
          for (int instance1 : instances)
          {
            for (int instance2 : instances)
            {
              matrix(instance1, instance2) += 1;
            }
          }
        }
        break;
      }

      Node currentNode = nodes.front();
      nodes.pop_front();

      vector<int> nodeIndices = currentNode.indices;
      vector<int> nodeInstances = currentNode.instances;
      vector<int> left_indices, right_indices, left_instances, right_instances;
      int colNum = performSplit(data, attributes_indices, attributes, left_indices, right_indices, nodeInstances);
      if (colNum == 0)
      {
      for (int i : left_indices) {
        left_instances.push_back(nodeInstances[i]);
      }
      for (int i : right_indices) {
        right_instances.push_back(nodeInstances[i]);
      }

      if (left_indices.size() < nmin and left_indices.size() != 0)
      {
        for (int instance1 : left_instances)
        {
          for (int instance2 : left_instances)
          {
            matrix(instance1, instance2) += left_instances.size();
          }
        }
      }

      else
      {
        Node currentNode = {left_indices, left_instances};
        nodes.push_back(currentNode);
      }

      if (right_indices.size() < nmin and right_indices.size() != 0)
      {
        for (int instance1 : right_instances)
        {
          for (int instance2 : right_instances)
          {
            matrix(instance1, instance2) += right_instances.size();
          }
        }
      }

      else
      {
        Node currentNode = {right_indices, right_instances};
        nodes.push_back(currentNode);
      }
    }

        for (int instance1 : left_instances) {
            for (int instance2 : right_instances) {
              matrix(instance2,instance1) = matrix(instance1,instance2) += (left_instances.size() + right_instances.size());

        }
    }
  }
  }
  return matrix/(nrows*nTrees);
}

vector<vector<int>> readCSV(string filename, char sep)
{
    ifstream dataFile;
    dataFile.open(filename);
    vector<vector<int>> csv;
    while (!dataFile.eof())
    {
        string line;
        getline(dataFile, line, '\n');
        stringstream buffer(line);
        string tmp;
        vector<int> values;

        while (getline(buffer, tmp, sep))
        {
            values.push_back(strtod(tmp.c_str(), 0));
        }
        csv.push_back(values);
    }

    return csv;
}

int main(int argc, char *argv[])
{

    vector<vector<int>> data;
    int nTrees = 200;
    vector<vector<int>> j;
    if (argc == 3) {
      j = readCSV(argv[1], *argv[2]);
    }
    else {
      j = readCSV(argv[1], '\t');

    }
    vector<int> labels;
    int nrows = j.size();
    cout << nrows << endl;
    int nmin = 0;

    data = j;
    data.pop_back();
    const auto startTime = high_resolution_clock::now();
    
    MatrixXd matrix = build_randomized_tree_and_get_sim(data, nmin, nTrees);
    const auto endTime = high_resolution_clock::now();
    cout << matrix(1,1) << endl;
    printf("Time: %fms\n", duration_cast<duration<double, milli>>(endTime - startTime).count());


    ofstream fichier("./matrix.csv", ios::out | ios::trunc);
    for (int i = 0; i < nrows-1; i++)
    {
        for (int j = 0; j < nrows-1; j++)
        {
            fichier << matrix(i, j) << '\t';
        }
        fichier << '\n';
    }
    fichier.close();

    return 1;
}
