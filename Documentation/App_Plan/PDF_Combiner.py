from pypdf import PdfMerger
import os

def merge_pdfs(input_pdf_paths, output_filename):
    """
    Merges multiple PDF files into a single PDF.

    Args:
        input_pdf_paths (list): A list of paths to the PDF files to merge.
        output_filename (str): The name of the output merged PDF file.
    """
    merger = PdfMerger()

    for pdf_path in input_pdf_paths:
        try:
            merger.append(pdf_path)
        except Exception as e:
            print(f"Error appending {pdf_path}: {e}")

    try:
        merger.write(output_filename)
        print(f"PDFs successfully merged into {output_filename}")
    except Exception as e:
        print(f"Error writing merged PDF: {e}")
    finally:
        merger.close()

if __name__ == "__main__":
    # Example usage:
    # Create some dummy PDF files for testing (you would use your actual files)
    # import PyPDF2 # For creating dummy files if needed
    # writer = PyPDF2.PdfFileWriter()
    # writer.addBlankPage(width=612, height=792) # A4 size
    # with open("file1.pdf", "wb") as f:
    #     writer.write(f)
    # with open("file2.pdf", "wb") as f:
    #     writer.write(f)

    #get folder path
    folder_path = "path/to/your/pdf/folder"  # Replace with the actual folder path
    #parse folder to get list of pdf files to merge
    # List of PDF files to merge
    pdf_files_to_merge = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')
    ]
    # Output filename
    merged_pdf_name = "combined_document.pdf"

    merge_pdfs(pdf_files_to_merge, merged_pdf_name)