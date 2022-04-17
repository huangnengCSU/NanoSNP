#include "../common/cpp_aux.hpp"
#include "../common/line_reader.hpp"

#include <set>
#include <string>
#include <vector>
#include <iostream>

using namespace std;

void
extract_char_name(string& line, string& name)
{
    name.clear();
    for (auto c : line) {
        if (isspace(c)) break;
        name += c;
    }
}

int main(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr, "USAGE:\n");
        fprintf(stderr, "%s <pileup_file> <output_dir> <chr1> [...]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* pileup_path = argv[1];
    const char* output_dir = argv[2];
    set<string> dumpped_chr_set;
    for (int i = 3; i < argc; ++i) dumpped_chr_set.insert(string(argv[i]));

    LineReader* line_reader = new LineReader(pileup_path);
    string line;
    bool dump_this_chr_pileup = false;
    string last_chr_name;
    string chr_name;
    FILE* out = NULL;
    string path;
    int cnt = 0;

    while (line_reader->getline(line)) {
        if (line.empty()) continue;
        //cerr << line << '\n';
        extract_char_name(line, chr_name);
        if (chr_name != last_chr_name) {
            if (out) {
                fprintf(stderr, "    Dump %10d pileup data for %s\n", cnt, last_chr_name.c_str());
                safe_fclose(out); 
                out = nullptr;
            }
            cnt = 0;
            last_chr_name = chr_name;
            if (dumpped_chr_set.find(chr_name) != dumpped_chr_set.end()) {
                dump_this_chr_pileup = true;
                fprintf(stderr, "Dump pileup data for %s\n", chr_name.c_str());
                path = output_dir;
                if (path.size() && path.back() != '/') path += '/';
                path += chr_name;
                path += ".mpileup";
                out = safe_fopen(path.c_str(), "w");
            } else {
                dump_this_chr_pileup = false;
            }
        }
        
        if (!dump_this_chr_pileup) continue;
        safe_fwrite(line.c_str(), 1, line.size(), out);
        fprintf(out, "\n");
        ++cnt;
        //if ((cnt % 10000) == 0) fprintf(stderr, "%10d pileup data dummped\n", cnt);
    }
    delete line_reader;

    if (out) {
        fprintf(stderr, "    Dump %10d pileup data for %s\n", cnt, last_chr_name.c_str());
        safe_fclose(out); 
        out = nullptr;        
    }
}