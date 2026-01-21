import csv
from faker import Faker

# Initialize Faker
fake = Faker()

# Define the number of records
num_records = 1000
filename = "cmpnydta.csv"

# Write data to the CSV file
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Writing the header row
    writer.writerow(["Company Name", "Employee Name"])

    # Generating and writing 1000 records
    for _ in range(num_records):
        company = fake.company()
        name = fake.name()
        writer.writerow([company, name])

print(f"Successfully generated {filename} with {num_records} records.")
