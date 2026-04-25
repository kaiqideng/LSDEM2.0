#pragma once
#include "CUDAKernelFunction/myUtility/myVec.h"
#include <algorithm>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <vector_types.h>

class LSGridNodeInfo
{
public:
    LSGridNodeInfo() = default;
    
    void buildLSGrid(const int resolutionPerParticleDiameter = 50)
    {
        if (!isValid()) return;
        if (resolutionPerParticleDiameter <= 0) return;

        const double3 boxMin = boundingBoxMin();
        const double3 boxMax = boundingBoxMax();

        const double boxSizeX = boxMax.x - boxMin.x;
        const double boxSizeY = boxMax.y - boxMin.y;
        const double boxSizeZ = boxMax.z - boxMin.z;

        const double particleReferenceDiameter = std::max(boxSizeX,
        std::max(boxSizeY, boxSizeZ));

        if (particleReferenceDiameter <= 0.0) return;

        spacing_ = particleReferenceDiameter / double(resolutionPerParticleDiameter);
        buildLSGridKernel();
    };

    void reverseSDFSign()
    {
        for (double& v : SFD_) v = -v;
    }

    void outputGridVTI(const std::string& filename, const int argc, char** argv) const
    {
        if (SFD_.empty()) return;
        if (size3D_.x <= 0 || size3D_.y <= 0 || size3D_.z <= 0) return;

                const char* argv0 = (argc > 0 && argv != nullptr) ? argv[0] : nullptr;

        std::filesystem::path path(filename);
        if (!path.is_absolute())
        {
            std::filesystem::path exePath =
                (argv0 != nullptr)
                ? std::filesystem::weakly_canonical(std::filesystem::absolute(argv0))
                : std::filesystem::current_path();

            path = exePath.parent_path() / path;
        }

        std::ofstream fout(path.string() + ".vti");
        if (!fout.is_open()) return;

        fout << "<?xml version=\"1.0\"?>\n";
        fout << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";

        fout << "<ImageData ";
        fout << "Origin=\""
            << origin_.x << " "
            << origin_.y << " "
            << origin_.z << "\" ";

        fout << "Spacing=\""
            << spacing_ << " "
            << spacing_ << " "
            << spacing_ << "\" ";

        fout << "WholeExtent=\""
            << 0 << " " << size3D_.x - 1 << " "
            << 0 << " " << size3D_.y - 1 << " "
            << 0 << " " << size3D_.z - 1 << "\">\n";

        fout << "<Piece Extent=\""
            << 0 << " " << size3D_.x - 1 << " "
            << 0 << " " << size3D_.y - 1 << " "
            << 0 << " " << size3D_.z - 1 << "\">\n";

        fout << "<PointData Scalars=\"SDF\">\n";
        fout << "<DataArray type=\"Float64\" Name=\"SDF\" format=\"ascii\">\n";

        for (int z = 0; z < size3D_.z; ++z)
        {
            for (int y = 0; y < size3D_.y; ++y)
            {
                for (int x = 0; x < size3D_.x; ++x)
                {
                    const int idx = linearIndex3D(make_int3(x,y,z), size3D_);
                    fout << SFD_[idx] << " ";
                }
                fout << "\n";
            }
            fout << "\n";
        }

        fout << "</DataArray>\n";
        fout << "</PointData>\n";

        fout << "</Piece>\n";
        fout << "</ImageData>\n";
        fout << "</VTKFile>\n";

        fout.close();
    }

    const double3& origin() const { return origin_; }
    const int3& size3D() const { return size3D_; }
    double spacing() const { return spacing_; }
    const std::vector<double>& SFD() const { return SFD_; }
    const std::vector<double3>& boundaryNodePosition() const { return boundaryNodePosition_; }
    const std::vector<int3>& boundaryNodeConnectivity() const { return boundaryNodeConnectivity_; }

protected:
    void buildImplicitBoundaryNode(const int subdivisionLevel)
    {
        boundaryNodePosition_.clear();
        boundaryNodeConnectivity_.clear();

        // ---------- helpers ----------
        auto makeEdgeKey = [](int a, int b)
        {
            uint32_t minv = std::min(a,b);
            uint32_t maxv = std::max(a,b);
            return (uint64_t(minv) << 32) | uint64_t(maxv);
        };

        // ---------- 1. Icosahedron ----------
        const double t = (1.0 + std::sqrt(5.0)) * 0.5;

        boundaryNodePosition_ = {
            make_double3(-1, t, 0), make_double3(1, t, 0),
            make_double3(-1,-t, 0), make_double3(1,-t, 0),

            make_double3(0,-1, t), make_double3(0,1, t),
            make_double3(0,-1,-t), make_double3(0,1,-t),

            make_double3(t,0,-1), make_double3(t,0,1),
            make_double3(-t,0,-1), make_double3(-t,0,1)
        };

        boundaryNodeConnectivity_ = {
            {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
            {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
            {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
            {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}
        };

        // ---------- project initial vertices ----------
        for (auto& v : boundaryNodePosition_)
        {
            v = normalize(v);
            v = project2Surface(v);
        }

        // ---------- 2. subdivision ----------
        for (int level = 0; level < subdivisionLevel; ++level)
        {
            std::unordered_map<uint64_t,int> cache;
            std::vector<int3> newTriangles;
            newTriangles.reserve(boundaryNodeConnectivity_.size() * 4);

            auto getMid = [&](int a, int b)
            {
                uint64_t key = makeEdgeKey(a,b);
                auto it = cache.find(key);
                if (it != cache.end()) return it->second;

                double3 mid = 0.5 * (boundaryNodePosition_[a] + boundaryNodePosition_[b]);
                mid = project2Surface(mid);

                int idx = (int)boundaryNodePosition_.size();
                boundaryNodePosition_.push_back(mid);
                cache[key] = idx;
                return idx;
            };

            for (const auto& tri : boundaryNodeConnectivity_)
            {
                int a = tri.x;
                int b = tri.y;
                int c = tri.z;

                int ab = getMid(a,b);
                int bc = getMid(b,c);
                int ca = getMid(c,a);

                newTriangles.push_back(make_int3(a,ab,ca));
                newTriangles.push_back(make_int3(b,bc,ab));
                newTriangles.push_back(make_int3(c,ca,bc));
                newTriangles.push_back(make_int3(ab,bc,ca));
            }

            boundaryNodeConnectivity_ = std::move(newTriangles);
        }
    }

    std::vector<double3> boundaryNodePosition_;
    std::vector<int3> boundaryNodeConnectivity_;

private:
    virtual bool isValid() const = 0;
    virtual double3 boundingBoxMin() const = 0;
    virtual double3 boundingBoxMax() const = 0;
    virtual double evaluateSFD(const double3& point) const = 0;
    virtual double3 project2Surface(const double3& point) const = 0;

    void buildLSGridKernel()
    {
        const double3 boxMin = boundingBoxMin();
        const double3 boxMax = boundingBoxMax();
        const double paddingDistance = 2.0 * spacing_;

        origin_ = make_double3(boxMin.x - paddingDistance,
                                boxMin.y - paddingDistance,
                                boxMin.z - paddingDistance);
        origin_ = make_double3(std::floor(origin_.x / spacing_) * spacing_,
                                std::floor(origin_.y / spacing_) * spacing_,
                                std::floor(origin_.z / spacing_) * spacing_);
        double3 paddedBoxMax = make_double3(boxMax.x + paddingDistance,
                                                boxMax.y + paddingDistance,
                                                boxMax.z + paddingDistance);
        paddedBoxMax = make_double3(std::ceil(paddedBoxMax.x / spacing_) * spacing_,
                                        std::ceil(paddedBoxMax.y / spacing_) * spacing_,
                                        std::ceil(paddedBoxMax.z / spacing_) * spacing_);

        size3D_.x = static_cast<int>(std::floor((paddedBoxMax.x - origin_.x) / spacing_) + 0.5) + 1;
        size3D_.y = static_cast<int>(std::floor((paddedBoxMax.y - origin_.y) / spacing_) + 0.5) + 1;
        size3D_.z = static_cast<int>(std::floor((paddedBoxMax.z - origin_.z) / spacing_) + 0.5) + 1;

        SFD_.resize(size3D_.x * size3D_.y * size3D_.z, 0.0);
        for (int z = 0; z < size3D_.z; ++z)
        {
            for (int y = 0; y < size3D_.y; ++y)
            {
                for (int x = 0; x < size3D_.x; ++x)
                {
                    const double3 point = make_double3(origin_.x + x * spacing_,
                                                        origin_.y + y * spacing_,
                                                        origin_.z + z * spacing_);
                    SFD_[linearIndex3D(make_int3(x, y, z), size3D_)] = evaluateSFD(point);
                }
            }
        }
    }

    double3 origin_{0.,0.,0.};
    int3 size3D_{0,0,0};
    double spacing_{0};
    std::vector<double> SFD_;
};

class TriangleMesh : public LSGridNodeInfo
{
public:
    TriangleMesh() = default;

    TriangleMesh(const std::vector<double3>& vertexPosition, const std::vector<int3>& triangleVertexID)
    {
        boundaryNodePosition_ = vertexPosition;
        boundaryNodeConnectivity_ = triangleVertexID;
        if (isValid()) buildBVH();
    }

    void loadOBJ(const std::string& filename, const int argc, char** argv)
    {
        boundaryNodePosition_.clear();
        boundaryNodeConnectivity_.clear();
        BVHNodes_.clear();
        triRefs_.clear();

        const char* argv0 = (argc > 0 && argv != nullptr) ? argv[0] : nullptr;

        std::filesystem::path path(filename);
        if (!path.is_absolute())
        {
            std::filesystem::path exePath =
                (argv0 != nullptr)
                ? std::filesystem::weakly_canonical(std::filesystem::absolute(argv0))
                : std::filesystem::current_path();

            path = exePath.parent_path() / path;
        }

        std::ifstream fin(path.string());
        if (!fin.is_open()) return;

        std::string line;
        size_t lineNumber = 0;

        auto trim = [](const std::string& s)
        {
            const auto first = s.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) return std::string();
            const auto last = s.find_last_not_of(" \t\r\n");
            return s.substr(first, last - first + 1);
        };

        auto startsWith = [](const std::string& s, const std::string& prefix)
        {
            return s.size() >= prefix.size() &&
                std::equal(prefix.begin(), prefix.end(), s.begin());
        };

        try
        {
            while (std::getline(fin, line))
            {
                ++lineNumber;
                line = trim(line);

                if (line.empty() || line[0] == '#') continue;

                // -------------------------
                // vertex
                // -------------------------
                if (startsWith(line, "v "))
                {
                    std::istringstream iss(line);
                    std::string tag;
                    double x, y, z;

                    iss >> tag >> x >> y >> z;
                    if (iss.fail()) 
                    {
                        boundaryNodePosition_.clear();
                        boundaryNodeConnectivity_.clear();
                        return;
                    }

                    boundaryNodePosition_.push_back(make_double3(x,y,z));
                }

                // -------------------------
                // face
                // -------------------------
                else if (startsWith(line, "f "))
                {
                    std::istringstream iss(line);
                    std::string tag;
                    iss >> tag;

                    std::vector<int> face;
                    std::string token;

                    while (iss >> token)
                    {
                        size_t slash = token.find('/');
                        std::string idxStr = (slash == std::string::npos)
                                            ? token
                                            : token.substr(0, slash);

                        if (idxStr.empty()) return;

                        int objIndex = std::stoi(idxStr);

                        int idx;
                        if (objIndex > 0)
                            idx = objIndex - 1;
                        else if (objIndex < 0)
                            idx = (int)boundaryNodePosition_.size() + objIndex;
                        else
                        {
                            boundaryNodePosition_.clear();
                            boundaryNodeConnectivity_.clear();
                            return;
                        }

                        if (idx < 0 || idx >= (int)boundaryNodePosition_.size())
                        {
                            boundaryNodePosition_.clear();
                            boundaryNodeConnectivity_.clear();
                            return;
                        }

                        face.push_back(idx);
                    }

                    if (face.size() < 3) 
                    {
                        boundaryNodePosition_.clear();
                        boundaryNodeConnectivity_.clear();
                        return;
                    }

                    // triangle
                    if (face.size() == 3)
                    {
                        boundaryNodeConnectivity_.push_back(
                            make_int3(face[0], face[1], face[2]));
                    }
                    else
                    {
                        // fan triangulation
                        for (size_t i = 1; i + 1 < face.size(); ++i)
                        {
                            boundaryNodeConnectivity_.push_back(
                                make_int3(face[0], face[i], face[i+1]));
                        }
                    }
                }
            }
        }
        catch (...)
        {
            boundaryNodePosition_.clear();
            boundaryNodeConnectivity_.clear();
            return;
        }

        if (isValid()) buildBVH();
    }

private:
    bool isValid() const override
    {
        return !boundaryNodePosition_.empty() && !boundaryNodeConnectivity_.empty();
    }

    double3 boundingBoxMin() const override
    {
        double3 bmin = make_double3(1e30,1e30,1e30);
        for (const auto& v : boundaryNodePosition_)
            bmin = minVec(bmin, v);
        return bmin;
    }

    double3 boundingBoxMax() const override
    {
        double3 bmax = make_double3(-1e30,-1e30,-1e30);
        for (const auto& v : boundaryNodePosition_)
            bmax = maxVec(bmax, v);
        return bmax;
    }

    double evaluateSFD(const double3& p) const override
    {
        if (BVHNodes_.empty()) return 0.0;

        double d = queryDistance(p, 0, 1e30);

        const double3 dir = make_double3(1.0, 0.123, 0.456);
        int count = rayCount(p, dir, 0);

        return (count & 1) ? -d : d;
    }

private:
    struct BVHNode
    {
        double3 bmin, bmax;
        int left = -1;
        int right = -1;
        int start = 0;
        int count = 0;
        bool isLeaf() const { return count > 0; }
    };

    struct TriangleRef
    {
        int id;
        double3 centroid;
    };

    std::vector<TriangleRef> triRefs_;
    std::vector<BVHNode> BVHNodes_;

private:
    void buildBVH()
    {
        triRefs_.resize(boundaryNodeConnectivity_.size());

        for (int i = 0; i < (int)boundaryNodeConnectivity_.size(); ++i)
        {
            const auto& tri = boundaryNodeConnectivity_[i];
            const double3& v0 = boundaryNodePosition_[tri.x];
            const double3& v1 = boundaryNodePosition_[tri.y];
            const double3& v2 = boundaryNodePosition_[tri.z];

            double3 c = make_double3(
                (v0.x + v1.x + v2.x) / 3.0,
                (v0.y + v1.y + v2.y) / 3.0,
                (v0.z + v1.z + v2.z) / 3.0
            );

            triRefs_[i] = { i, c };
        }

        BVHNodes_.clear();
        if (!triRefs_.empty())
            buildBVHNode(0, triRefs_.size());
    }

    int buildBVHNode(int start, int end)
    {
        BVHNode node;
        computeBounds(node.bmin, node.bmax, start, end);

        int n = end - start;

        if (n <= 8)
        {
            node.start = start;
            node.count = n;
            BVHNodes_.push_back(node);
            return (int)BVHNodes_.size() - 1;
        }

        double3 extent = make_double3(
            node.bmax.x - node.bmin.x,
            node.bmax.y - node.bmin.y,
            node.bmax.z - node.bmin.z
        );

        int axis = (extent.x > extent.y && extent.x > extent.z) ? 0 :
                   (extent.y > extent.z) ? 1 : 2;

        int mid = (start + end) / 2;

        std::nth_element(triRefs_.begin() + start,
                         triRefs_.begin() + mid,
                         triRefs_.begin() + end,
                         [axis](const TriangleRef& a, const TriangleRef& b)
        {
            return (&a.centroid.x)[axis] < (&b.centroid.x)[axis];
        });

        int idx = BVHNodes_.size();
        BVHNodes_.push_back(node);

        int left = buildBVHNode(start, mid);
        int right = buildBVHNode(mid, end);

        BVHNodes_[idx].left = left;
        BVHNodes_[idx].right = right;

        return idx;
    }

    void computeBounds(double3& bmin, double3& bmax, int start, int end) const
    {
        bmin = make_double3(1e30,1e30,1e30);
        bmax = make_double3(-1e30,-1e30,-1e30);

        for (int i = start; i < end; ++i)
        {
            const auto& tri = boundaryNodeConnectivity_[triRefs_[i].id];

            const double3& v0 = boundaryNodePosition_[tri.x];
            const double3& v1 = boundaryNodePosition_[tri.y];
            const double3& v2 = boundaryNodePosition_[tri.z];

            bmin = minVec(bmin, v0);
            bmin = minVec(bmin, v1);
            bmin = minVec(bmin, v2);

            bmax = maxVec(bmax, v0);
            bmax = maxVec(bmax, v1);
            bmax = maxVec(bmax, v2);
        }
    }

    double queryDistance(const double3& p, int nodeIdx, double best) const
    {
        const BVHNode& node = BVHNodes_[nodeIdx];

        double dbox = distanceToAABB(p, node.bmin, node.bmax);
        if (dbox > best) return best;

        if (node.isLeaf())
        {
            for (int i = 0; i < node.count; ++i)
            {
                int triID = triRefs_[node.start + i].id;
                const auto& tri = boundaryNodeConnectivity_[triID];

                double d = pointTriangleDistance(
                    p,
                    boundaryNodePosition_[tri.x],
                    boundaryNodePosition_[tri.y],
                    boundaryNodePosition_[tri.z]);

                if (d < best) best = d;
            }
            return best;
        }

        best = queryDistance(p, node.left, best);
        best = queryDistance(p, node.right, best);

        return best;
    }

    int rayCount(const double3& o, const double3& d, int nodeIdx) const
    {
        const BVHNode& node = BVHNodes_[nodeIdx];

        if (!rayIntersectAABB(o, d, node.bmin, node.bmax))
            return 0;

        if (node.isLeaf())
        {
            int cnt = 0;
            for (int i = 0; i < node.count; ++i)
            {
                int triID = triRefs_[node.start + i].id;
                const auto& tri = boundaryNodeConnectivity_[triID];

                if (rayIntersectTriangle(o, d,
                    boundaryNodePosition_[tri.x],
                    boundaryNodePosition_[tri.y],
                    boundaryNodePosition_[tri.z]))
                    cnt++;
            }
            return cnt;
        }

        return rayCount(o, d, node.left)
             + rayCount(o, d, node.right);
    }

private:
    static double3 minVec(const double3& a, const double3& b)
    {
        return make_double3(std::min(a.x,b.x),
                            std::min(a.y,b.y),
                            std::min(a.z,b.z));
    }

    static double3 maxVec(const double3& a, const double3& b)
    {
        return make_double3(std::max(a.x,b.x),
                            std::max(a.y,b.y),
                            std::max(a.z,b.z));
    }

    static double distanceToAABB(const double3& p,
                                 const double3& bmin,
                                 const double3& bmax)
    {
        double dx = std::max(std::max(bmin.x - p.x, 0.0), p.x - bmax.x);
        double dy = std::max(std::max(bmin.y - p.y, 0.0), p.y - bmax.y);
        double dz = std::max(std::max(bmin.z - p.z, 0.0), p.z - bmax.z);
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    static bool rayIntersectAABB(const double3& o, const double3& d,
                                const double3& bmin, const double3& bmax)
    {
        double tmin = (bmin.x - o.x) / d.x;
        double tmax = (bmax.x - o.x) / d.x;
        if (tmin > tmax) std::swap(tmin, tmax);

        double tymin = (bmin.y - o.y) / d.y;
        double tymax = (bmax.y - o.y) / d.y;
        if (tymin > tymax) std::swap(tymin, tymax);

        if ((tmin > tymax) || (tymin > tmax)) return false;

        tmin = std::max(tmin, tymin);
        tmax = std::min(tmax, tymax);

        double tzmin = (bmin.z - o.z) / d.z;
        double tzmax = (bmax.z - o.z) / d.z;
        if (tzmin > tzmax) std::swap(tzmin, tzmax);

        if ((tmin > tzmax) || (tzmin > tmax)) return false;

        return true;
    }

    static double pointTriangleDistance(const double3& p,
                                        const double3& a,
                                        const double3& b,
                                        const double3& c)
    {
        double3 ab = b - a;
        double3 ac = c - a;
        double3 ap = p - a;

        double d1 = dot(ab, ap);
        double d2 = dot(ac, ap);

        if (d1 <= 0 && d2 <= 0) return length(ap);

        double3 bp = p - b;
        double d3 = dot(ab, bp);
        double d4 = dot(ac, bp);
        if (d3 >= 0 && d4 <= d3) return length(bp);

        double vc = d1*d4 - d3*d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0)
        {
            double v = d1 / (d1 - d3);
            double3 proj = make_double3(a.x + v*ab.x,
                                        a.y + v*ab.y,
                                        a.z + v*ab.z);
            return length(p - proj);
        }

        double3 cp = p - c;
        double d5 = dot(ab, cp);
        double d6 = dot(ac, cp);
        if (d6 >= 0 && d5 <= d6) return length(cp);

        double vb = d5*d2 - d1*d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0)
        {
            double w = d2 / (d2 - d6);
            double3 proj = make_double3(a.x + w*ac.x,
                                        a.y + w*ac.y,
                                        a.z + w*ac.z);
            return length(p - proj);
        }

        double va = d3*d6 - d5*d4;
        if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0)
        {
            double w = (d4-d3) / ((d4-d3)+(d5-d6));
            double3 bc = c - b;
            double3 proj = make_double3(b.x + w*bc.x,
                                        b.y + w*bc.y,
                                        b.z + w*bc.z);
            return length(p - proj);
        }

        double3 n = cross(ab, ac);
        return std::abs(dot(ap, n)) / length(n);
    }

    static bool rayIntersectTriangle(const double3& o, const double3& d,
                                     const double3& v0,
                                     const double3& v1,
                                     const double3& v2)
    {
        double3 v0v1 = v1 - v0;
        double3 v0v2 = v2 - v0;

        double3 pvec = cross(d, v0v2);
        double det = dot(v0v1, pvec);

        if (isZero(det)) return false;

        double invDet = 1.0 / det;

        double3 tvec = o - v0;
        double u = dot(tvec, pvec) * invDet;
        if (u < 0 || u > 1) return false;

        double3 qvec = cross(tvec, v0v1);
        double v = dot(d, qvec) * invDet;
        if (v < 0 || u + v > 1) return false;

        double t = dot(v0v2, qvec) * invDet;

        return !isZero(t);
    }
};

class Sphere : public TriangleMesh
{
public:
    Sphere() = default;

    Sphere(const double radius)
    {
        if (radius <= 0.0) return;
        radius_ = radius;
    }

    void setRadius(const double radius)
    {
        if (radius <= 0.0) return;
        radius_ = radius;
    }

    void buildBoundaryNode(const int subdivisionLevel = 4)
    {
        buildImplicitBoundaryNode(subdivisionLevel);
    }

private:
    bool isValid() const override
    {
        return radius_ > 0.0;
    }

    double3 boundingBoxMin() const override
    {
        return make_double3(-radius_, -radius_, -radius_);
    }

    double3 boundingBoxMax() const override
    {
        return make_double3(radius_, radius_, radius_);
    }

    double evaluateSFD(const double3& p) const override
    {
        double d = length(p - boundaryNodePosition_[0]);
        return d - radius_;
    }

    double3 project2Surface(const double3& p) const override
    {
        double3 dir = p;
        double len = length(dir);
        if (!isZero(len)) return radius_ * dir / len;
        else return make_double3(0, 0, 0);
    }

    double radius_{0.};
};

class Superellipsoid : public TriangleMesh
{
public:
    Superellipsoid() = default;

    Superellipsoid(const double rx,
                   const double ry,
                   const double rz,
                   const double ee,
                   const double en)
    {
        if (rx <= 0.0 || ry <= 0.0 || rz <= 0.0 ||
            ee <= 0.0 || en <= 0.0) return;

        rx_ = rx;
        ry_ = ry;
        rz_ = rz;
        ee_ = ee;
        en_ = en;
    }

    void setParameter(const double rx,
                    const double ry,
                    const double rz,
                    const double ee,
                    const double en)
    {
        if (rx <= 0.0 || ry <= 0.0 || rz <= 0.0 ||
            ee <= 0.0 || en <= 0.0) return;

        rx_ = rx;
        ry_ = ry;
        rz_ = rz;
        ee_ = ee;
        en_ = en;
    }

    void buildBoundaryNode(const int subdivisionLevel = 4)
    {
        buildImplicitBoundaryNode(subdivisionLevel);
    }

private:
    bool isValid() const override
    {
        return (rx_ > 0.0 &&
                ry_ > 0.0 &&
                rz_ > 0.0 &&
                ee_ > 0.0 &&
                en_ > 0.0);
    }

    double3 boundingBoxMin() const override
    {
        return make_double3(-rx_, -ry_, -rz_);
    }

    double3 boundingBoxMax() const override
    {
        return make_double3(rx_, ry_, rz_);
    }

    double evaluateSFD(const double3& p) const override
    {
        const double invRx = 1.0 / rx_;
        const double invRy = 1.0 / ry_;
        const double invRz = 1.0 / rz_;

        const double ax = std::fabs(p.x * invRx);
        const double ay = std::fabs(p.y * invRy);
        const double az = std::fabs(p.z * invRz);

        const double expXY = 2.0 / ee_;
        const double expZ  = 2.0 / en_;
        const double outer = ee_ / en_;

        const double xTerm = std::pow(ax, expXY);
        const double yTerm = std::pow(ay, expXY);

        const double xy = xTerm + yTerm;

        const double xyBlock = std::pow(xy, outer);
        const double zBlock  = std::pow(az, expZ);

        return xyBlock + zBlock - 1.0;
    }

    double3 project2Surface(const double3& p) const override
    {
        double3 x = p;

        for (int i = 0; i < 3; ++i)
        {
            const double invRx = 1.0 / rx_;
            const double invRy = 1.0 / ry_;
            const double invRz = 1.0 / rz_;

            const double ax = std::fabs(x.x * invRx);
            const double ay = std::fabs(x.y * invRy);
            const double az = std::fabs(x.z * invRz);

            const double expXY = 2.0 / ee_;
            const double expZ  = 2.0 / en_;
            const double outer = ee_ / en_;

            const double eps = 1e-30;

            const double safe_ax = std::max(ax, eps);
            const double safe_ay = std::max(ay, eps);
            const double safe_az = std::max(az, eps);

            const double xTerm = std::pow(safe_ax, expXY);
            const double yTerm = std::pow(safe_ay, expXY);
            const double xy = xTerm + yTerm;

            const double safe_xy = std::max(xy, eps);
            const double xyPow = std::pow(safe_xy, outer - 1.0);

            const double dxy_dx =
                expXY *
                std::pow(safe_ax, expXY - 1.0) *
                (x.x >= 0.0 ? 1.0 : -1.0) * invRx;

            const double dxy_dy =
                expXY *
                std::pow(safe_ay, expXY - 1.0) *
                (x.y >= 0.0 ? 1.0 : -1.0) * invRy;

            const double dx =
                outer * xyPow * dxy_dx;

            const double dy =
                outer * xyPow * dxy_dy;

            const double dz =
                expZ *
                std::pow(safe_az, expZ - 1.0) *
                (x.z >= 0.0 ? 1.0 : -1.0) * invRz;

            const double3 grad = make_double3(dx, dy, dz);

            const double phi = std::pow(xy, outer) + std::pow(az, expZ) - 1.0;

            const double g2 = dot(grad, grad);
            if (g2 < 1e-30) break;

            x = x - phi * grad / g2;
        }

        return x;
    }

    double rx_{1.0};
    double ry_{1.0};
    double rz_{1.0};
    double ee_{1.0};
    double en_{1.0};
};