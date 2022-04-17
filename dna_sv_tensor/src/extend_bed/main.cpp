#include "../common/line_reader.hpp"
#include "../common/cpp_aux.hpp"

#include <vector>

using namespace std;

void
extend_chr_bed_list(vector<pair<int, int>>& bed_list, const int extend_size, const char* chr_name, FILE* out)
{
    fprintf(stderr, "extend bed list for %s\n", chr_name);
    const int n_bed = bed_list.size();
    int i = 0;
    while (i < n_bed) {
        int left = bed_list[i].first - extend_size;
        left = max(0, left);
        int right = bed_list[i].second + extend_size;
        int j = i + 1;
        while (j < n_bed) {
            if (right < bed_list[j].first - extend_size) break;
            right = bed_list[j].second + extend_size;
            ++j;
        }

#if 0
        fprintf(stderr, "Merge\n");
        for (int k = i; k < j; ++k) {
            fprintf(stderr, "%10d\t%10d\n", bed_list[k].first, bed_list[k].second);
        }
        fprintf(stderr, "into %10d\t%10d\n", left, right);
#endif

        fprintf(out, "%s\t%d\t%d\n", chr_name, left, right);
        i = j;
    }    
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        fprintf(stderr, "USAGE:\n");
        fprintf(stderr, "%s bed_path extend_size output\n", argv[0]);
        return 1;
    }
    const char* bed_path = argv[1];
    const int extend_size = atoi(argv[2]);
    const char* output_path = argv[3];
    FILE* out = safe_fopen(output_path, "w");

    string bed_line;
    vector<string> bed_components;
    LineReader* line_reader = new LineReader(bed_path);
    vector<pair<int, int>> bed_list;
    string last_chr_name;

    while (line_reader->getline(bed_line)) {
        if (bed_line.empty()) continue;
        if (bed_line[0] == '#') continue;
        bed_components.clear();
        split_line(bed_line, "\t", bed_components);
        string& chr_name = bed_components[0];
        if (chr_name != last_chr_name) {
            if (!bed_list.empty()) extend_chr_bed_list(bed_list, extend_size, last_chr_name.c_str(), out);
            last_chr_name = chr_name;
            bed_list.clear();
        }
        hbn_assert(bed_components.size() >= 3);
        pair<int, int> bed;
        bed.first = atoi(bed_components[1].c_str());
        bed.second = atoi(bed_components[2].c_str());
        bed_list.push_back(bed);        
    }   
    if (!bed_list.empty()) extend_chr_bed_list(bed_list, extend_size, last_chr_name.c_str(), out);
    delete line_reader;

    return 0;
}