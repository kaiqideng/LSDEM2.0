// NOTE:
// Most of the code on this page was generated with the assistance of AI tools.

#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <vector_types.h>
#include <vector_functions.h>
#include <filesystem>

inline std::filesystem::path getExecutableDirectory(const char* argv0)
{
    if (argv0 == nullptr) return std::filesystem::current_path();

    std::filesystem::path exePath(argv0);

    if (exePath.is_relative())
    {
        exePath = std::filesystem::absolute(exePath);
    }

    exePath = std::filesystem::weakly_canonical(exePath);
    return exePath.parent_path();
}

inline std::string resolvePathFromExecutable(const std::string& filename,
                                             const char* argv0)
{
    const std::filesystem::path p(filename);

    if (p.is_absolute())
    {
        return p.string();
    }

    return (getExecutableDirectory(argv0) / p).string();
}

namespace OBJLoader
{
    //=========================================================================
    // String utilities
    //=========================================================================

    inline std::string trim(const std::string& s)
    {
        const auto first = s.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return "";

        const auto last = s.find_last_not_of(" \t\r\n");
        return s.substr(first, last - first + 1);
    }

    inline bool startsWith(const std::string& s, const std::string& prefix)
    {
        return s.size() >= prefix.size() &&
               std::equal(prefix.begin(), prefix.end(), s.begin());
    }

    //=========================================================================
    // OBJ face index parsing
    //=========================================================================
    // Supported face token examples:
    //   "12"
    //   "12/3/9"
    //   "12//9"
    //   "12/3"
    //
    // Only the vertex index is extracted here.
    // The returned index is still the original OBJ index and is not converted
    // to zero-based C++ indexing yet.
    inline int parseOBJVertexIndex(const std::string& token)
    {
        if (token.empty())
        {
            throw std::runtime_error("Empty face token in OBJ.");
        }

        const size_t slashPos = token.find('/');
        const std::string indexStr = (slashPos == std::string::npos)
            ? token
            : token.substr(0, slashPos);

        if (indexStr.empty())
        {
            throw std::runtime_error("Invalid OBJ face token: " + token);
        }

        return std::stoi(indexStr);
    }

    //=========================================================================
    // OBJ index conversion
    //=========================================================================
    // OBJ indexing rules:
    //   Positive index :  1, 2, 3, ...
    //   Negative index : -1 means the last previously defined vertex
    //
    // This function converts OBJ indexing into zero-based C++ indexing.
    inline int convertOBJIndexToZeroBased(const int objIndex,
                                          const int numVertices)
    {
        if (objIndex > 0)
        {
            return objIndex - 1;
        }
        else if (objIndex < 0)
        {
            return numVertices + objIndex;
        }
        else
        {
            throw std::runtime_error("OBJ index cannot be zero.");
        }
    }

    //=========================================================================
    // OBJ mesh loader
    //=========================================================================
    // Supported records:
    //   v x y z
    //   f i j k
    //
    // Notes:
    //   1. Only vertex positions and face connectivity are loaded.
    //   2. Face tokens such as i/j/k, i//k, and i/j are supported.
    //   3. Polygonal faces with more than 3 vertices are triangulated using
    //      a fan triangulation scheme.
    inline bool loadOBJMesh(const std::string& filename,
                        std::vector<double3>& vertices,
                        std::vector<int3>& faces,
                        const int argc,
                        char** argv,
                        const bool verbose = true)
    {
        vertices.clear();
        faces.clear();

        const char* argv0 = (argc > 0 && argv != nullptr) ? argv[0] : nullptr;
        const std::string resolvedFilename =
            resolvePathFromExecutable(filename, argv0);

        std::ifstream fin(resolvedFilename);
        if (!fin.is_open())
        {
            std::cerr << "[OBJLoader] Failed to open file: "
                    << resolvedFilename << std::endl;
            return false;
        }

        std::string line;
        size_t lineNumber = 0;

        try
        {
            while (std::getline(fin, line))
            {
                ++lineNumber;
                line = trim(line);

                if (line.empty() || line[0] == '#')
                {
                    continue;
                }

                //-----------------------------------------------------------------
                // Vertex record
                //-----------------------------------------------------------------
                if (startsWith(line, "v "))
                {
                    std::istringstream iss(line);

                    std::string tag;
                    double x, y, z;
                    iss >> tag >> x >> y >> z;

                    if (iss.fail())
                    {
                        throw std::runtime_error(
                            "Failed to parse vertex at line " +
                            std::to_string(lineNumber));
                    }

                    vertices.push_back(make_double3(x, y, z));
                }

                //-----------------------------------------------------------------
                // Face record
                //-----------------------------------------------------------------
                else if (startsWith(line, "f "))
                {
                    std::istringstream iss(line);

                    std::string tag;
                    iss >> tag;

                    std::vector<int> polygonVertexIndices;
                    std::string token;

                    while (iss >> token)
                    {
                        const int objIndex = parseOBJVertexIndex(token);

                        const int zeroBasedIndex =
                            convertOBJIndexToZeroBased(
                                objIndex,
                                static_cast<int>(vertices.size()));

                        if (zeroBasedIndex < 0 ||
                            zeroBasedIndex >= static_cast<int>(vertices.size()))
                        {
                            throw std::runtime_error(
                                "Face index out of range at line " +
                                std::to_string(lineNumber));
                        }

                        polygonVertexIndices.push_back(zeroBasedIndex);
                    }

                    if (polygonVertexIndices.size() < 3)
                    {
                        throw std::runtime_error(
                            "Face has fewer than 3 vertices at line " +
                            std::to_string(lineNumber));
                    }

                    //-----------------------------------------------------------------
                    // Triangle face
                    //-----------------------------------------------------------------
                    if (polygonVertexIndices.size() == 3)
                    {
                        faces.push_back(make_int3(
                            polygonVertexIndices[0],
                            polygonVertexIndices[1],
                            polygonVertexIndices[2]));
                    }

                    //-----------------------------------------------------------------
                    // Polygon face: fan triangulation
                    //-----------------------------------------------------------------
                    else
                    {
                        for (size_t i = 1; i + 1 < polygonVertexIndices.size(); ++i)
                        {
                            faces.push_back(make_int3(
                                polygonVertexIndices[0],
                                polygonVertexIndices[i],
                                polygonVertexIndices[i + 1]));
                        }
                    }
                }

                //-----------------------------------------------------------------
                // Ignore other records such as:
                //   vn, vt, g, usemtl, o, ...
                //-----------------------------------------------------------------
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "[OBJLoader] Error while reading " << resolvedFilename
                    << " at line " << lineNumber
                    << ": " << e.what() << std::endl;

            vertices.clear();
            faces.clear();
            return false;
        }

        if (verbose)
        {
            std::cout << "[OBJLoader] Loaded OBJ: " << resolvedFilename << std::endl;
            std::cout << "  Num vertices = " << vertices.size() << std::endl;
            std::cout << "  Num faces    = " << faces.size() << std::endl;
        }

        return true;
    }
}