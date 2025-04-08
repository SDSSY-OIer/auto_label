#include <string>
#include <vector>
#include <filesystem>
using namespace std;
namespace fs = filesystem;
constexpr std::size_t N = 200;
int main()
{
    const fs::path folder_path = "./img";

    // 递归记录指定文件夹下jpg文件路径
    vector<fs::path> jpgs;
    for (const auto &path : fs::recursive_directory_iterator(folder_path))
    {
        if (path.is_regular_file() && path.path().extension() == ".jpg")
        {
            jpgs.emplace_back(path);
        }
    }

    // 创建临时文件夹
    std::size_t sz = 0;
    for (const auto &path : fs::directory_iterator(folder_path))
    {
        if (path.is_directory())
        {
            sz = max(sz, path.path().filename().string().size());
        }
    }
    fs::path temp_path = folder_path / string(++sz, 't');
    fs::create_directories(temp_path);

    // 将jpg文件转移至临时文件夹下并重命名
    for (std::size_t i = 0; i < jpgs.size(); ++i)
    {
        fs::path new_path = temp_path / (to_string(i + 1) + ".jpg");
        fs::rename(jpgs[i], new_path);
        jpgs[i] = new_path;
    }

    // 记录指定文件夹下其余文件路径
    vector<fs::path> junkfiles, junkfolders;
    for (const auto &path : fs::directory_iterator(folder_path))
    {
        if (path.is_regular_file())
        {
            junkfiles.emplace_back(path);
        }
        else if (path.is_directory() && path != temp_path)
        {
            junkfolders.emplace_back(path);
        }
    }

    // 删除其余文件
    for (const auto &path : junkfiles)
    {
        fs::remove(path);
    }
    for (const auto &path : junkfolders)
    {
        fs::remove_all(path);
    }

    // 创建打包文件夹
    for (std::size_t i = 1; i <= (jpgs.size() + N - 1) / N; ++i)
    {
        fs::create_directories(folder_path / fs::path(to_string(i)));
    }

    // 将jpg文件放入对应文件夹
    for (std::size_t i = 0; i < jpgs.size(); ++i)
    {
        fs::rename(jpgs[i], folder_path / fs::path(to_string(i / N + 1)) / jpgs[i].filename());
    }

    // 删除临时文件夹
    fs::remove_all(temp_path);
}