# coding=utf-8

# -*- coding: utf-8 -*-

import requests
import xlsxwriter
import html
import re


headers = {"content-language": "vi-VN"}

def html_to_text(value):
    if not isinstance(value, str):
        return value
    # Giải mã HTML entities
    value = html.unescape(value)
    # Loại bỏ thẻ HTML
    value = re.sub(r'<[^>]+>', '', value)
    return value.strip()

def create_sheet(workbook, sheet_name):
    worksheet = workbook.add_worksheet(sheet_name)

    # Ghi thông tin vào worksheet
    worksheet.write('A1', 'Tên sản phẩm')
    worksheet.write('B1', 'Khối lượng')
    worksheet.write('C1', 'Giá')
    worksheet.write('D1', 'Mô tả')
    worksheet.write('E1', 'Mô tả đủ')
    worksheet.write('F1', 'Hướng dẫn sử dụng')
    worksheet.write('G1', 'Thành phần')
    worksheet.write('H1', 'Hình ảnh')
    worksheet.write('I1', 'Link')
    worksheet.write('J1', 'Thương hiệu')

    return worksheet

def write_data_to_sheet(worksheet, products):
    row = 1
    for product in products:
        url = "https://pyapi.greenwaystart.com/pyapi/v1/greenway/shop/product/" + product["code"] + "/"
        response = requests.get(url, headers=headers)
        product_detail = response.json()
        brand_code = product_detail.get("brand", {}).get("path", "")
        product_url = f"https://greenwayglobal.vn/shop/brands/{brand_code}/{product['code']}"
        
        print("Product Detail:", product_detail)
        
        worksheet.write(row, 0, html_to_text(product_detail.get("name", "")))
        worksheet.write(row, 1, html_to_text(product_detail.get("tech_info", "")))
        worksheet.write(row, 2, html_to_text(product_detail["stock_product"]["price"]))
        worksheet.write(row, 3, html_to_text(product_detail.get("short_description", "")))                      
        worksheet.write(row, 4, html_to_text(product_detail.get("content", "")))
        worksheet.write(row, 5, html_to_text(product_detail.get("application", "")))
        worksheet.write(row, 6, html_to_text(product_detail.get("composition", "")))
        worksheet.write(row, 7, html_to_text(product_detail["images"][0]["origin"]["path"]))
        worksheet.write(row, 8, product_url)
        worksheet.write(row, 9, html_to_text(html_to_text(product_detail.get("brand", {}).get("name", ""))))
        row += 1

# Gọi API
url = "https://pyapi.greenwaystart.com/pyapi/v1/greenway/shop/"
response = requests.get(url, headers=headers)
data = response.json()

# Tạo file Excel mới với xlsxwriter
workbook = xlsxwriter.Workbook('subsections_xlsxwriter.xlsx')
dict_sections = set()

for section in data["sections"]:
    subsections = section.get("subsections", [])
    for subsection in subsections:
               # Sheet name tối đa 31 ký tự, loại bỏ ký tự đặc biệt
        print("Subsection:", subsection)
        safe_sheet_name = subsection["name"].replace('TPCN', 'Thực phẩm chức năng')
        safe_sheet_name = safe_sheet_name[:30]
        if (safe_sheet_name not in dict_sections):
            dict_sections.add(safe_sheet_name)
            worksheet = create_sheet(workbook, safe_sheet_name)

            url = "https://pyapi.greenwaystart.com/pyapi/v1/greenway/shop/section/"  + section['path'] + "/" + subsection['path'] + "/"
            print("URL:", url)
            response = requests.get(url, headers=headers)
            data = response.json()
            products = data.get("products", []) 
            write_data_to_sheet(worksheet, products)

# lấy promotions
url = "https://pyapi.greenwaystart.com/pyapi/v1/greenway/shop/product/promo/"
data = requests.get(url, headers=headers).json()
if data:
    worksheet = create_sheet(workbook, "Khuyến mãi")
    products = data.get("sales", [])
    write_data_to_sheet(worksheet, products)

workbook.close()
print("Đã lưu mỗi subsection vào 1 sheet trong subsections_xlsxwriter.xlsx")