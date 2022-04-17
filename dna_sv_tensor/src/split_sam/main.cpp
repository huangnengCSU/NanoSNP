#include "../common/cpp_aux.hpp"
#include "../common/line_reader.hpp"

#include <map>

using namespace std;

void
dump_usage(const char* pn)
{
    fprintf(stderr, "\n");
    fprintf(stderr, "USAGE:\n");
    fprintf(stderr, "%s -sam <sorted_sam_file> -output_dir <dir_for_splited_sam> -chr <chr_name>\n", pn);
}

void
extract_ref_name_from_sam(std::string& sam, string& ref_name)
{
    ref_name.clear();

    size_t n = sam.size();
    size_t i = 0;
    while (i < n && sam[i] != '\t') ++i;
    ++i;
    while (i < n && sam[i] != '\t') ++i;
    ++i;

    if (i >= n) return;

    while (i < n && sam[i] != '\t') {
        ref_name += sam[i];
        ++i;
    }
}

void
extract_ref_name_from_sam(const char* sam, string& ref_name)
{
    ref_name.clear();

    const char* p = sam;
    while (*p != '\0' && *p != '\t') ++p;
    ++p;
    while (*p != '\0' && *p != '\t') ++p;
    ++p;

    while (*p != '\0' && *p != '\t') {
        ref_name += *p;
        ++p;
    }    
}

int main(int argc, char* argv[])
{
    const char* sorted_sam_file = "-";
    const char* output_dir = ".";
    const char* chr_name = nullptr;

    int i = 1;
    while (i < argc) {
        if (strcmp(argv[i], "-sam") == 0) {
            sorted_sam_file = argv[i+1];
            i += 2;
            continue;
        } else if (strcmp(argv[i], "-output_dir") == 0) {
            output_dir = argv[i+1];
            i += 2;
            continue;
        } else if (strcmp(argv[i], "-chr") == 0) {
            chr_name = argv[i + 1];
            i += 2;
            continue;
        }
        fprintf(stderr, "Unrecognised option '%s'\n", argv[i]);
        dump_usage(argv[0]);
        return 1;
    }
    create_directory(output_dir);

    vector<string> hdr_list;
    map<string, string> ref_map;
    string sam_line;
    vector<string> sam_components;
    bool end_of_file = false;
    LineReader* line_reader = new LineReader(sorted_sam_file);

    while (1) {
        if (!line_reader->getline(sam_line)) {
            end_of_file = true;
            break;
        }
        if (sam_line.empty()) continue;
        if (sam_line[0] != '@') break;
        sam_components.clear();
        split_line(sam_line, "\t", sam_components);
        bool is_ref_info = (sam_components.size() > 1) && (strncmp(sam_components[1].c_str(), "SN:", 3) == 0);
        if (is_ref_info) {
            ref_map[sam_components[1].substr(3)] = sam_line;
        } else {
            hdr_list.push_back(sam_line);
        }
    }
    if (end_of_file) {
        delete line_reader;
        return EXIT_SUCCESS;
    }

    string ref_sam_path;
    string last_ref_name;
    string ref_name;
    FILE* out = NULL;
    bool find_chr_sam = false;
    int sam_cnt = 0;

    while (1) {
        if (!line_reader->getline(sam_line)) break;
        extract_ref_name_from_sam(sam_line.c_str(), ref_name);
        if (ref_name == "*") continue;
        if (ref_name != last_ref_name) {
            if (sam_cnt) {
                hbn_assert(!last_ref_name.empty());
                fprintf(stderr, "%10d SAM records dumpped for %s\n", sam_cnt, last_ref_name.c_str());
                sam_cnt = 0;
            }
            if (find_chr_sam) break;
            if (out) safe_fclose(out);
            last_ref_name = ref_name;
            if (chr_name && ref_name != chr_name) continue;
            fprintf(stderr, "Dump SAM results for %s\n", ref_name.c_str());
            if (chr_name) find_chr_sam = true;
            ref_sam_path = output_dir;
            if (ref_sam_path.back() != '/') ref_sam_path += '/';
            ref_sam_path += ref_name;
            ref_sam_path += ".sam";
            out = safe_fopen(ref_sam_path.c_str(), "w");
            for (auto& hdr : hdr_list) {
                safe_fwrite(hdr.c_str(), 1, hdr.size(), out);
                fprintf(out, "\n");
            }
            auto pos = ref_map.find(ref_name);
            hbn_assert(pos != ref_map.end(), "fail to find reference name %s in sam line \n%s", ref_name.c_str(), sam_line.c_str());  
            safe_fwrite(pos->second.c_str(), 1, pos->second.size(), out);
            fprintf(out, "\n");        
        }
        if (chr_name && ref_name != chr_name) continue;
        safe_fwrite(sam_line.c_str(), 1, sam_line.size(), out);
        fprintf(out, "\n");
        ++sam_cnt;
        //if ((sam_cnt % 10000) == 0) HBN_LOG("%10d SAM (%s) processed", sam_cnt, last_ref_name.c_str());
    }
    if (sam_cnt) {
        hbn_assert(!last_ref_name.empty());
        fprintf(stderr, "%10d SAM records dumpped for %s\n", sam_cnt, last_ref_name.c_str());
        sam_cnt = 0;
    }
    safe_fclose(out);
    delete line_reader;
}