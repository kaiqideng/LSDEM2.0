#pragma once
#include <string>

//static void ltrim(std::string& s)
//{
//    size_t i = 0;
//    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
//    s.erase(0, i);
//}
//
//static bool getlineValid(std::ifstream& fin, std::string& line)
//{
//    while (std::getline(fin, line))
//    {
//        ltrim(line);
//        if (line.empty() || line[0] == '#') continue;
//        return true;
//    }
//    return false;
//}
//
//static int countValidNumbersStrict(const std::string& line)
//{
//    std::istringstream iss(line);
//    std::string token;
//    int count = 0;
//    while (iss >> token)
//    {
//        std::istringstream tokStream(token);
//        double d;
//        char c;
//        if (tokStream >> d && !(tokStream >> c))
//        {
//            ++count;
//        }
//    }
//    return count;
//}

#ifdef _WIN32
#include <io.h>
#include <direct.h>                 // _mkdir

#define MKDIR(path) _mkdir(path)    // returns 0 on success, -1 if already exists

static inline int removeVtuFiles(const std::string& dir)
{
    std::string pattern = dir + "\\*.vtu";
    struct _finddata_t fdata;
    intptr_t h = _findfirst(pattern.c_str(), &fdata);
    if (h == -1) return 0;

    int removed = 0;
    do {
        std::string full = dir + "\\" + fdata.name;
        if (std::remove(full.c_str()) == 0) ++removed;
    } while (_findnext(h, &fdata) == 0);
    _findclose(h);
    return removed;
}

static inline int removeDatFiles(const std::string& dir)
{
    std::string pattern = dir + "\\*.dat";
    struct _finddata_t fdata;
    intptr_t h = _findfirst(pattern.c_str(), &fdata);
    if (h == -1) return 0;

    int removed = 0;
    do {
        std::string full = dir + "\\" + fdata.name;
        if (std::remove(full.c_str()) == 0) ++removed;
    } while (_findnext(h, &fdata) == 0);
    _findclose(h);
    return removed;
}

#else
#include <dirent.h>
#include <sys/stat.h>               // mkdir
#include <cstring>

#define MKDIR(path) mkdir(path, 0755)

static bool inline hasVtuExt(const char* fname)
{
    const char* dot = strrchr(fname, '.');
    return dot && std::strcmp(dot, ".vtu") == 0;
}

static bool inline hasVtmExt(const char* fname)
{
    const char* dot = strrchr(fname, '.');
    return dot && std::strcmp(dot, ".vtm") == 0;
}

static bool inline hasVtiExt(const char* fname)
{
    const char* dot = strrchr(fname, '.');
    return dot && std::strcmp(dot, ".vti") == 0;
}

static bool inline hasDatExt(const char* fname)
{
    const char* dot = strrchr(fname, '.');
    return dot && std::strcmp(dot, ".dat") == 0;
}

static int inline removeVtuFiles(const std::string& dir)
{
    DIR* dp = opendir(dir.c_str());
    if (!dp) return 0;

    int removed = 0;
    struct dirent* ent;
    while ((ent = readdir(dp)) != NULL)
    {
        if (ent->d_type == DT_DIR) continue;
        if (!hasVtuExt(ent->d_name)) continue;

        std::string full = dir + "/" + ent->d_name;
        if (std::remove(full.c_str()) == 0) ++removed;
    }
    closedir(dp);
    return removed;
}

static int inline removeVtmFiles(const std::string& dir)
{
    DIR* dp = opendir(dir.c_str());
    if (!dp) return 0;

    int removed = 0;
    struct dirent* ent;
    while ((ent = readdir(dp)) != NULL)
    {
        if (ent->d_type == DT_DIR) continue;
        if (!hasVtmExt(ent->d_name)) continue;

        std::string full = dir + "/" + ent->d_name;
        if (std::remove(full.c_str()) == 0) ++removed;
    }
    closedir(dp);
    return removed;
}

static int inline removeVtiFiles(const std::string& dir)
{
    DIR* dp = opendir(dir.c_str());
    if (!dp) return 0;

    int removed = 0;
    struct dirent* ent;
    while ((ent = readdir(dp)) != NULL)
    {
        if (ent->d_type == DT_DIR) continue;
        if (!hasVtiExt(ent->d_name)) continue;

        std::string full = dir + "/" + ent->d_name;
        if (std::remove(full.c_str()) == 0) ++removed;
    }
    closedir(dp);
    return removed;
}

static int inline removeDatFiles(const std::string& dir)
{
    DIR* dp = opendir(dir.c_str());
    if (!dp) return 0;

    int removed = 0;
    struct dirent* ent;
    while ((ent = readdir(dp)) != NULL)
    {
        if (ent->d_type == DT_DIR) continue;
        if (!hasDatExt(ent->d_name)) continue;

        std::string full = dir + "/" + ent->d_name;
        if (std::remove(full.c_str()) == 0) ++removed;
    }
    closedir(dp);
    return removed;
}
#endif