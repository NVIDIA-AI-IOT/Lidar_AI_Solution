/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include "pango-cairo.hpp"

#ifdef ENABLE_TEXT_BACKEND_PANGO
#include "memory.hpp"
#include <pango/pangocairo.h>
#include <map>
#include <sstream>

using namespace std;

class PangoWordMeta : public WordMeta{
public:
    int w, h, yadvance, offset_x;
    unsigned long int word;

    virtual int width() const override{
        return w;
    }

    virtual int height() const override{
        return h;
    }

    virtual int xadvance(int font_size, bool empty) const override{
        if(empty){
            return font_size * 0.1;
        }
        return w;
    }

    virtual int x_offset_on_bitmap() const override{
        return offset_x;
    }

    PangoWordMeta() = default;
    PangoWordMeta(int w, int h, int yadvance, unsigned long int word, int offset_x) {
        this->w        = w;
        this->h        = h;
        this->yadvance = yadvance;
        this->word     = word;
        this->offset_x = offset_x;
    }
};

class PangoWordMetaMapperImpl : public WordMetaMapper, public map<unsigned long int, PangoWordMeta>{
public:
    virtual WordMeta* query(unsigned long int word) override{
        auto iter = this->find(word);
        if(iter == this->end()) return nullptr;
        return &iter->second;
    }
};

class PangoCairoBackend : public TextBackend{
private:
    unique_ptr<Memory<unsigned char>> text_bitmap;
    unique_ptr<Memory<unsigned char>> single_word_bitmap;
    map<string, PangoWordMetaMapperImpl> glyph_sets;
    map<string, vector<unsigned long int>> build_use_textes;
    int text_bitmap_width      = 0;
    int text_bitmap_height     = 0;
    int temp_size = 0;
    cairo_surface_t* single_word_surface = nullptr;
    cairo_t* cairo = nullptr;
    PangoLayout* layout = nullptr;
    bool has_new_text_need_build_bitmap = false;

public:
    PangoCairoBackend(){
        int temp_size   = MAX_FONT_SIZE * 2;
        this->temp_size = temp_size;
        this->single_word_bitmap.reset(new Memory<unsigned char>);
        this->single_word_bitmap->alloc_or_resize_to(temp_size * temp_size);
        this->single_word_surface = cairo_image_surface_create_for_data(this->single_word_bitmap->host(), CAIRO_FORMAT_A8, temp_size, temp_size, temp_size);
        this->cairo    = cairo_create (this->single_word_surface);
        this->layout   = pango_cairo_create_layout (this->cairo);
        pango_layout_set_wrap(this->layout, PANGO_WRAP_WORD);
        memset(this->single_word_bitmap->host(), 0, this->single_word_bitmap->bytes());
    }

    virtual ~PangoCairoBackend(){
        g_object_unref(this->layout);
        cairo_destroy(this->cairo);
        cairo_surface_destroy(this->single_word_surface);
    }

    virtual vector<unsigned long int> split_utf8(const char* utf8_text) override{
        vector<unsigned long int> output;
        output.reserve(strlen(utf8_text));

        unsigned char *str = (unsigned char *)utf8_text;
        unsigned int c = 0;
        while (*str) {
            if (!(*str & 0x80))
                output.emplace_back(((unsigned int)*str++) | (1ul << 32));
            else if ((*str & 0xe0) == 0xc0) {
                if (*str < 0xc2) return {};
                c = *str++;
                if ((*str & 0xc0) != 0x80) return {};
                output.emplace_back(c | ((unsigned int)*str++ << 8) | (2ul << 32));
            } else if ((*str & 0xf0) == 0xe0) {
                if (*str == 0xe0 && (str[1] < 0xa0 || str[1] > 0xbf)) return {};
                if (*str == 0xed && str[1] > 0x9f) return {}; // str[1] < 0x80 is checked below
                c = *str++;
                if ((*str & 0xc0) != 0x80) return {};
                c |= (unsigned int)*str++ << 8;
                if ((*str & 0xc0) != 0x80) return {};
                output.emplace_back(c | (*str++ << 16) | (3ul << 32));
            } else if ((*str & 0xf8) == 0xf0) {
                if (*str > 0xf4) return {};
                if (*str == 0xf0 && (str[1] < 0x90 || str[1] > 0xbf)) return {};
                if (*str == 0xf4 && str[1] > 0x8f) return {}; // str[1] < 0x80 is checked below
                unsigned int c0 = (*str++ & 0x07) << 18;
                c = *str;
                if ((*str & 0xc0) != 0x80) return {};
                c0 += (*str++ & 0x3f) << 12;
                c |= (unsigned int)*str++ << 8;
                if ((*str & 0xc0) != 0x80) return {};
                c0 += (*str++ & 0x3f) << 6;
                c |= (unsigned int)*str++ << 16;
                if ((*str & 0xc0) != 0x80) return {};
                c0 += (*str++ & 0x3f);
                c |= (unsigned int)*str++ << 24;
                // utf-8 encodings of values used in surrogate pairs are invalid
                if ((c0 & 0xFFFFF800) == 0xD800) return {};
                if (c0 >= 0x10000) {
                    output.emplace_back(c | (4ul << 32));
                }
            } else
                return {};
        }
        return output;
    }

    virtual std::tuple<int, int, int> measure_text(const std::vector<unsigned long int>& words, unsigned int font_size, const char* font) override{
        int draw_x = 0;
        int xadvance = font_size * 0.1;
        int max_h = 0;
        auto font_and_size = concat_font_name_size(font, font_size);

        auto& word_map = this->glyph_sets[font_and_size];
        for (auto& word : words) {
            int w, h;
            auto iter = word_map.find(word);
            if(iter == word_map.end()){
                PangoFontDescription* desc = pango_font_description_from_string(font_and_size.c_str());
                pango_layout_set_font_description (this->layout, desc);
                pango_font_description_free (desc);

                int length = word >> 32;
                pango_layout_set_text (this->layout, (const char*)&word, length);
                pango_cairo_update_layout (this->cairo, this->layout);
                pango_layout_get_pixel_size (this->layout, &w, &h);
            }else{
                w = iter->second.w;
                h = iter->second.h;
            }

            if (w < 1 || h < 1) {
                draw_x += xadvance;
            } else {
                draw_x += w;
            }
            max_h = max(max_h, h);
        }
        return make_tuple(draw_x, max_h, 0);
    }

    virtual void add_build_text(const std::vector<unsigned long int>& words, unsigned int font_size, const char* font) override{

        auto font_and_size = concat_font_name_size(font, font_size);
        auto& maps = build_use_textes[font_and_size];
        auto& glyph_map = this->glyph_sets[font_and_size];
        for (auto& word : words) {
            if (glyph_map.find(word) != glyph_map.end()) continue;
            maps.insert(maps.end(), word);
            has_new_text_need_build_bitmap = true;
        }
    }

    virtual WordMetaMapper* query(const char* font, int font_size) override{
        auto font_and_size = concat_font_name_size(font, font_size);
        auto iter = this->glyph_sets.find(font_and_size);
        if(iter == this->glyph_sets.end()) return nullptr;
        return &iter->second;
    }

    virtual void build_bitmap(void* _stream) override{

        cudaStream_t stream       = (cudaStream_t)_stream;

        // 1. collect all word shape.
        if (!has_new_text_need_build_bitmap){
            has_new_text_need_build_bitmap = false;
            build_use_textes.clear();
            return;
        }

        for (auto& textes : build_use_textes) {

            auto& glyph_map = this->glyph_sets[textes.first];
            auto& words     = textes.second;
            bool desc_set = false;
            for (auto& word : words) {
                if (glyph_map.find(word) != glyph_map.end()) continue;

                if(!desc_set){
                    desc_set = true;
                    PangoFontDescription* desc = pango_font_description_from_string(textes.first.c_str());
                    pango_layout_set_font_description (this->layout, desc);
                    pango_font_description_free (desc);
                }

                int w, h;
                int length = word >> 32;
                pango_layout_set_text (this->layout, (const char*)&word, length);
                pango_cairo_update_layout (this->cairo, this->layout);
                pango_layout_get_pixel_size (this->layout, &w, &h);
                glyph_map.insert(make_pair(word, PangoWordMeta(w, h, 0, word, 0)));
            }

            const char* default_words = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:-&./&^%$#@!+=\\[];,'\"?` ";
            const char* pword = default_words;
            for (; *pword; ++pword) {
                unsigned long int word = (unsigned int)*pword | (1ul << 32);
                if (glyph_map.find(word) != glyph_map.end()) continue;

                int w, h;
                int length = word >> 32;
                pango_layout_set_text (this->layout, (const char*)&word, length);
                pango_cairo_update_layout (this->cairo, this->layout);
                pango_layout_get_pixel_size (this->layout, &w, &h);
                glyph_map.insert(make_pair(word, PangoWordMeta(w, h, 0, word, 0)));
            }
        }

        int max_glyph_height  = 0;
        int max_glyph_width   = 0;
        int total_glyph_width = 0;
        for (auto& map : this->glyph_sets) {
            for(auto& item : map.second){
                int w = item.second.w, h = item.second.h;
                max_glyph_width    = std::max(max_glyph_width, w);
                max_glyph_height   = std::max(max_glyph_height, h);
                total_glyph_width += w;
            }
        }

        if (this->text_bitmap   == nullptr) this->text_bitmap.reset(new Memory<unsigned char>());
        this->text_bitmap_width  = total_glyph_width;
        this->text_bitmap_height = max_glyph_height;
        this->text_bitmap->alloc_or_resize_to(total_glyph_width * max_glyph_height);
        memset(this->text_bitmap->host(), 0, this->text_bitmap->bytes());

        // Rasterize word to bitmap
        int offset_x = 0;
        for (auto& map : this->glyph_sets) {
            PangoFontDescription* desc = pango_font_description_from_string(map.first.c_str());
            pango_layout_set_font_description (this->layout, desc);
            pango_font_description_free (desc);

            for(auto& item : map.second){
                auto& glyph = item.second;

                int w = glyph.w, h = glyph.h;
                if (w < 1 || h < 1) continue;

                glyph.offset_x = offset_x;
                int length = glyph.word >> 32;
                memset(this->single_word_bitmap->host(), 0, this->single_word_bitmap->bytes());
                pango_layout_set_text (this->layout, (const char*)&glyph.word, length);
                pango_cairo_update_layout (this->cairo, this->layout);
                cairo_move_to (this->cairo, 0, 0);
                pango_cairo_show_layout (this->cairo, this->layout);

                unsigned char* prow = this->text_bitmap->host() + offset_x;
                unsigned char* psrc = this->single_word_bitmap->host();
                for(int iy = 0; iy < h; ++iy, prow += this->text_bitmap_width, psrc += this->temp_size){
                    memcpy(prow, psrc, w);
                }
                offset_x += w;
            }
        }
        this->text_bitmap->copy_host_to_device(stream);
        this->has_new_text_need_build_bitmap = false;
        this->build_use_textes.clear();
    }

    virtual unsigned char* bitmap_device_pointer() const override{
        if(!this->text_bitmap) return nullptr;
        return this->text_bitmap->device();
    }

    virtual int bitmap_width() const override{
        return this->text_bitmap_width;
    }

    virtual int compute_y_offset(int max_glyph_height, int h, WordMeta* word, int font_size) const override{
        (void)word;
        (void)font_size;
        return max_glyph_height - h;
    }

    virtual int uniform_font_size(int size) const override{
        return size;
    }
};

std::shared_ptr<TextBackend> create_pango_cairo_backend(){
    return std::make_shared<PangoCairoBackend>();
}
#endif // ENABLE_TEXT_BACKEND_PANGO