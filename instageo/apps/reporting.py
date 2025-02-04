import os
import numpy as np
import xarray as xr
import plotly.io as pio  # For saving Plotly figures as images
import plotly.graph_objects as go
from email.mime.image import MIMEImage  # For attaching images to emails


def generate_high_density_report(
    fig: go.Figure, rasters: dict[str, xr.Dataset], threshold=0.8
):
    """Generates a report of locations with density above the given threshold."""
    report = []
    for tile_filename, raster in rasters.items():
        try:
            data_array = raster["band_data"]

            if np.any(
                data_array.to_numpy() > threshold
            ):  # this maybe will be gone with folium fig
                country_code = os.path.basename(tile_filename).split("_")[1][1:]
                report.append({"country_code": country_code, "filepath": tile_filename})

        except Exception as e:
            print(f"Error processing {tile_filename}: {e}")  # Or use a proper logger

    map_image_path = "temp_map.png"
    pio.write_image(fig, map_image_path)
    return report, map_image_path


def format_report(report, map_image_path):
    if not report:
        return "No high-density areas detected."

    formatted_report = "High-density areas detected:\n"
    for entry in report:
        formatted_report += (
            f"- Country: {entry['country_code']}, File: {entry['filepath']}\n"
        )

    formatted_report += (
        f"<img src='cid:{map_image_path}' width='500'>\n"  # width reduced
    )

    return formatted_report


def send_email(to_email, subject, body, img_path):
    """Sends an email. Replace with your preferred email sending method."""
    try:
        import smtplib
        from email.mime.text import MIMEText

        from_mail = "locustbusters@gmail.com"
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = from_mail
        msg["To"] = to_email
        with open(img_path, "rb") as img_file:  # Open in binary read mode
            img = MIMEImage(img_file.read())
            img.add_header(
                "Content-ID", f"<{img_path}>"
            )  # Important for referencing in HTML
            msg.attach(img)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_mail, "rvabulmpjdtuwvza")
            server.send_message(msg)

        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
