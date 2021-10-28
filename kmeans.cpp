#include "kmeans.hpp"

#include <cassert>
#include <chrono>
#include <limits>
#include <queue>
#include <thread>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;

inline double Point::Distance(const Point &other) const noexcept {
  double a = x - other.x;
  double b = y - other.y;
  return a * a + b * b;
}

std::istream &operator>>(std::istream &is, Point &pt) {
  return is >> pt.x >> pt.y;
}

std::ostream &operator<<(std::ostream &os, Point &pt) {
  return os << pt.x << " " << pt.y;
}

Kmeans::Kmeans(const std::vector<Point> &points,
               const std::vector<Point> &init_centers) {
  m_points = points;
  m_centers = init_centers;
  m_numPoints = points.size();
  m_numCenters = init_centers.size();
}

std::vector<index_t> Kmeans::Run(int max_iterations) {
  std::vector<index_t> assignment(m_numPoints, 0);  // the return vector
  int curr_iteration = 0;
  std::cout << "Running kmeans with num points = " << m_numPoints
            << ", num centers = " << m_numCenters
            << ", max iterations = " << max_iterations << "...\n";

  double loop_1 = 0, loop_2 = 0;

  std::vector<int> p_cnt(m_numCenters, 0);
  while (max_iterations--) {
    ++curr_iteration;
    bool changed = false;

#pragma omp parallel for
    for (int i = 0; i < m_numPoints; ++i) {
      Point &p_i = m_points[i];
      double min_dis = std::numeric_limits<double>::max();
      int ans = -1;
      for (int k = 0; k < m_numCenters; ++k) {
        Point &c_k = m_centers[k];
        double dis = p_i.Distance(c_k);
        if (dis < min_dis) {
          min_dis = dis;
          ans = k;
        }
      }
      if (ans != assignment[i]) {
        assignment[i] = ans;
        changed = true;
      }
    }

    if (!changed) {
      goto converge;
    }

    m_centers.assign(m_numCenters, Point());
    p_cnt.assign(m_numPoints, 0);

    int i;
    for (i = 0; i < m_numPoints - 3; i += 4) {
      index_t cluster_1 = assignment[i];
      index_t cluster_2 = assignment[i + 1];
      index_t cluster_3 = assignment[i + 2];
      index_t cluster_4 = assignment[i + 3];

      auto point_1_x = m_points[i].x, point_1_y = m_points[i].y;
      auto point_2_x = m_points[i + 1].x, point_2_y = m_points[i + 1].y;
      auto point_3_x = m_points[i + 2].x, point_3_y = m_points[i + 2].y;
      auto point_4_x = m_points[i + 3].x, point_4_y = m_points[i + 3].y;

      m_centers[cluster_1].x += point_1_x;
      m_centers[cluster_1].y += point_1_y;
      m_centers[cluster_2].x += point_2_x;
      m_centers[cluster_2].y += point_2_y;
      m_centers[cluster_3].x += point_3_x;
      m_centers[cluster_3].y += point_3_y;
      m_centers[cluster_4].x += point_4_x;
      m_centers[cluster_4].y += point_4_y;

      ++p_cnt[cluster_1];
      ++p_cnt[cluster_2];
      ++p_cnt[cluster_3];
      ++p_cnt[cluster_4];
    }

    for (; i < m_numPoints; ++i) {
      index_t cluster_1 = assignment[i];
      auto point_x = m_points[i].x, point_y = m_points[i].y;
      m_centers[cluster_1].x += point_x;
      m_centers[cluster_1].y += point_y;
      ++p_cnt[cluster_1];
    }

    for (int j = 0; j < m_numCenters; ++j) {
      m_centers[j].x /= p_cnt[j];
      m_centers[j].y /= p_cnt[j];
    }
  }

converge:
  std::cout << "Finished in " << curr_iteration << " iterations." << std::endl;
  return assignment;
}