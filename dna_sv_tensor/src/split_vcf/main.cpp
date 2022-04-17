#include "../common/cpp_aux.hpp"
#include "../common/line_reader.hpp"

#include <algorithm>
#include <map>
#include <set>

#include <cstdlib>
#include <cstdio>
#include <cstring>

using namespace std;

void
extract_genotype(const string& gt_str, int& gt1, int& gt2)
{
    string gts = gt_str.substr(0, gt_str.find(':'));
    for (auto& c : gts) if (c == '/') c = '|';
    for (auto& c : gts) if (c == '.') c = '0';
    auto bar_pos = gts.find('|');
    hbn_assert(bar_pos != string::npos);
    int t1 = atoi(gts.c_str());
    int t2 = atoi(gts.c_str() + bar_pos + 1);
    gt1 = min(t1, t2);
    gt2 = max(t1, t2);
}

static bool
fix_alternate(const string& vcf, string& alternate, int& gt1, int& gt2)
{
    auto start_pos = alternate.find('*');
    if (start_pos == string::npos) return true;

    int n_comma = 0;
    for (auto& c : alternate) if (c == ',') ++n_comma;
    if (gt1 + gt2 != 3 || n_comma != 1) {
        fprintf(stderr, "Invalid variant in vcf '%s' (alternate = %s, gt1 = %d, gt2 = %d)", vcf.c_str(), alternate.c_str(), gt1, gt2);
        return false;
    }
    gt1 = 0;
    gt2 = 1;
    size_t n = alternate.size(), m = 0;
    for (size_t i = 0; i < n; ++i) {
        if (alternate[i] != '*') alternate[m++] = alternate[i];
    }
    hbn_assert(m < n);
    alternate.resize(m);
    return true;
}

bool
is_indel_variant(const string& ref_bases, const string& alt_bases)
{
    int alt_l1 = alt_bases.size(), alt_l2 = alt_bases.size();

    auto comma_pos = alt_bases.find(',');
    if (comma_pos != string::npos) {
        alt_l1 = comma_pos;
        hbn_assert(alt_bases.size() > comma_pos + 1);
        alt_l2 = alt_bases.size() - comma_pos - 1;
    }

    return (alt_l1 != ref_bases.size()) || (alt_l2 != ref_bases.size());
}

void extract_true_variants(const char* vcf_path, const char* output_dir)
{
    LineReader* line_reader = new LineReader(vcf_path);
    string line;
    vector<string> columns;
    string last_chr_name;
    FILE* var_out = nullptr;
    FILE* vcf_out = nullptr;
    string output_path;
    int vcf_cnt = 0;

    while (line_reader->getline(line)) {
        if (line.empty() || line[0] == '#') continue;
        columns.clear();
        split_line(line, "\t", columns);
        const string& chr_name = columns[0];
        const string& chr_offset = columns[1];
        const string& chr_bases = columns[3];
        string& alt_bases = columns[4];
        const string& gt_str = columns.back();

        if (chr_name != last_chr_name) {
            if (vcf_cnt) fprintf(stderr, "\tDump %d VCF for %s\n", vcf_cnt, last_chr_name.c_str());
            vcf_cnt = 0;
            fprintf(stderr, "Dump VCF for %s\n", chr_name.c_str());
            if (var_out) fclose(var_out);
            output_path = output_dir;
            if (output_path.size() && output_path.back() != '/') output_path += '/';
            output_path += chr_name;
            output_path += ".true_var";
            last_chr_name = chr_name;
            var_out = fopen(output_path.c_str(), "w");

            if (vcf_out) fclose(vcf_out);
            output_path = output_dir;
            if (output_path.size() && output_path.back() != '/') output_path += '/';
            output_path += chr_name;
            output_path += ".vcf";
            vcf_out = fopen(output_path.c_str(), "w");          
        }

        int gt1 = 0, gt2 = 0;
        extract_genotype(gt_str, gt1, gt2);
        if (!fix_alternate(line, alt_bases, gt1, gt2)) continue;
        ++vcf_cnt;
        //if (!is_indel_variant(chr_bases, alt_bases)) continue;

        fprintf(var_out, "%s\t%s\t%s\t%s\t%d\t%d\n",
                chr_name.c_str(),
                chr_offset.c_str(),
                chr_bases.c_str(),
                alt_bases.c_str(),
                gt1,
                gt2);

        fprintf(vcf_out, "%s\n", line.c_str());
    }
    if (var_out) fclose(var_out);
    if (vcf_out) fclose(vcf_out);
    delete line_reader;
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        fprintf(stderr, "USAGE:\n");
        fprintf(stderr, "%s vcf_input output_dir\n", argv[0]);
        return 1;
    }
    const char* vcf_path = argv[1];
    const char* output_dir = argv[2];
    
    extract_true_variants(vcf_path, output_dir);

    return 0;
}