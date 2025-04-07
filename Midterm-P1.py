import csv
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
import tkinter as tk
from tkinter import messagebox

# Abstract Base Class for Plastic Waste Data
class GenPlasticWasteData:
    # Class-level attributes for plastic types and categories
    plastic_types = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]
    recyclable = ["PET", "HDPE"]
    non_recyclable = ["PVC"]
    sometimes_recyclable = ["LDPE", "PP", "PS", "Other"]

    def __init__(self, city, population):
        # Instance attributes for storing data
        self.city = city
        self.population = population
        self.recyclable_quantity = 0
        self.nonrecyclable_quantity = 0
        self.sometimesrecyclable_quantity = 0
        self.total_records = 0
        self.date_received = date.today()
        self.date_processed = None

    def generate_records(self):
        """This method needs to be implemented by the subclass."""
        raise NotImplementedError("This method must be implemented in a subclass.")

    def write_csv(self, file_path):
        """This method needs to be implemented by the subclass."""
        raise NotImplementedError("This method must be implemented in a subclass.")

    def read_csv(self, file_path):
        """This method needs to be implemented by the subclass."""
        raise NotImplementedError("This method must be implemented in a subclass.")

    def set_processed_date(self):
        """Sets the date when the data was processed."""
        self.date_processed = date.today()

    def calculate_total_records(self):
        """Calculates the total number of records based on the quantities."""
        self.total_records = (self.recyclable_quantity +
                              self.nonrecyclable_quantity +
                              self.sometimesrecyclable_quantity)

    # Methods to get the data for each category
    def get_city_recyclable_data(self):
        return self.recyclable_quantity

    def get_city_non_recyclable_data(self):
        return self.nonrecyclable_quantity

    def get_city_sometimes_recyclable_data(self):
        return self.sometimesrecyclable_quantity

    def get_date_received(self):
        return self.date_received

    def get_date_processed(self):
        return self.date_processed


# Subclass for Fremont
class FremontGenData(GenPlasticWasteData):
    def __init__(self, population):
        super().__init__("Fremont", population)

    def generate_data(self):
        super().generate_data()

    def generate_records(self):
        """Generates 10,000 random records for Fremont."""
        self.recyclable_quantity = 0
        self.nonrecyclable_quantity = 0
        self.sometimesrecyclable_quantity = 0

        for _ in range(10000):
            plastic_type = random.choice(self.plastic_types)
            quantity = random.randint(1, 100)

            if plastic_type in self.recyclable:
                self.recyclable_quantity += quantity
            elif plastic_type in self.non_recyclable:
                self.nonrecyclable_quantity += quantity
            elif plastic_type in self.sometimes_recyclable:
                self.sometimesrecyclable_quantity += quantity

        self.calculate_total_records()

    def write_csv(self, file_path):
        """Writes the generated data to the CSV file for Fremont."""
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["City", "Population", "Recyclable Quantity", "Non-Recyclable Quantity",
                             "Sometimes Recyclable Quantity", "Date Received", "Date Processed"])
            writer.writerow([self.city, self.population, self.recyclable_quantity,
                             self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                             self.date_received, self.date_processed])

    def read_csv(self, file_path):
        """Reads the CSV file and populates the Fremont data."""
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Type'] in self.recyclable:
                    self.recyclable_quantity += int(row['Quantity'])
                elif row['Type'] in self.non_recyclable:
                    self.nonrecyclable_quantity += int(row['Quantity'])
                elif row['Type'] in self.sometimes_recyclable:
                    self.sometimesrecyclable_quantity += int(row['Quantity'])
        self.calculate_total_records()


# Subclass for Hayward
class HaywardGenData(GenPlasticWasteData):
    def __init__(self, population):
        super().__init__("Hayward", population)

    def generate_data(self):
        super().generate_data()

    def generate_records(self):
        """Generates 10,000 random records for Hayward."""
        self.recyclable_quantity = 0
        self.nonrecyclable_quantity = 0
        self.sometimesrecyclable_quantity = 0

        for _ in range(10000):
            plastic_type = random.choice(self.plastic_types)
            quantity = random.randint(1, 100)

            if plastic_type in self.recyclable:
                self.recyclable_quantity += quantity
            elif plastic_type in self.non_recyclable:
                self.nonrecyclable_quantity += quantity
            elif plastic_type in self.sometimes_recyclable:
                self.sometimesrecyclable_quantity += quantity

        self.calculate_total_records()

    def write_csv(self, file_path):
        """Writes the generated data to the CSV file for Hayward."""
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["City", "Population", "Recyclable Quantity", "Non-Recyclable Quantity",
                             "Sometimes Recyclable Quantity", "Date Received", "Date Processed"])
            writer.writerow([self.city, self.population, self.recyclable_quantity,
                             self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                             self.date_received, self.date_processed])

    def read_csv(self, file_path):
        """Reads the CSV file and populates the Hayward data."""
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Type'] in self.recyclable:
                    self.recyclable_quantity += int(row['Quantity'])
                elif row['Type'] in self.non_recyclable:
                    self.nonrecyclable_quantity += int(row['Quantity'])
                elif row['Type'] in self.sometimes_recyclable:
                    self.sometimesrecyclable_quantity += int(row['Quantity'])
        self.calculate_total_records()


# Repeat similar classes for other cities (Oakland, Pleasanton, Dublin)
class OaklandGenData(GenPlasticWasteData):
    def __init__(self, population):
        super().__init__("Oakland", population)

    def generate_data(self):
        super().generate_data()

    def generate_records(self):
        """Generates 10,000 random records for Hayward."""
        self.recyclable_quantity = 0
        self.nonrecyclable_quantity = 0
        self.sometimesrecyclable_quantity = 0

        for _ in range(10000):
            plastic_type = random.choice(self.plastic_types)
            quantity = random.randint(1, 100)

            if plastic_type in self.recyclable:
                self.recyclable_quantity += quantity
            elif plastic_type in self.non_recyclable:
                self.nonrecyclable_quantity += quantity
            elif plastic_type in self.sometimes_recyclable:
                self.sometimesrecyclable_quantity += quantity

        self.calculate_total_records()

    def write_csv(self, file_path):
        """Writes the generated data to the CSV file for Hayward."""
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["City", "Population", "Recyclable Quantity", "Non-Recyclable Quantity",
                             "Sometimes Recyclable Quantity", "Date Received", "Date Processed"])
            writer.writerow([self.city, self.population, self.recyclable_quantity,
                             self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                             self.date_received, self.date_processed])

    def read_csv(self, file_path):
        """Reads the CSV file and populates the Hayward data."""
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Type'] in self.recyclable:
                    self.recyclable_quantity += int(row['Quantity'])
                elif row['Type'] in self.non_recyclable:
                    self.nonrecyclable_quantity += int(row['Quantity'])
                elif row['Type'] in self.sometimes_recyclable:
                    self.sometimesrecyclable_quantity += int(row['Quantity'])
        self.calculate_total_records()


class PleasantonGenData(GenPlasticWasteData):
    def __init__(self, population):
        super().__init__("Pleasanton", population)

    def generate_data(self):
        super().generate_data()

    def generate_records(self):
        """Generates 10,000 random records for Hayward."""
        self.recyclable_quantity = 0
        self.nonrecyclable_quantity = 0
        self.sometimesrecyclable_quantity = 0

        for _ in range(10000):
            plastic_type = random.choice(self.plastic_types)
            quantity = random.randint(1, 100)

            if plastic_type in self.recyclable:
                self.recyclable_quantity += quantity
            elif plastic_type in self.non_recyclable:
                self.nonrecyclable_quantity += quantity
            elif plastic_type in self.sometimes_recyclable:
                self.sometimesrecyclable_quantity += quantity

        self.calculate_total_records()

    def write_csv(self, file_path):
        """Writes the generated data to the CSV file for Hayward."""
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["City", "Population", "Recyclable Quantity", "Non-Recyclable Quantity",
                             "Sometimes Recyclable Quantity", "Date Received", "Date Processed"])
            writer.writerow([self.city, self.population, self.recyclable_quantity,
                             self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                             self.date_received, self.date_processed])

    def read_csv(self, file_path):
        """Reads the CSV file and populates the Hayward data."""
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Type'] in self.recyclable:
                    self.recyclable_quantity += int(row['Quantity'])
                elif row['Type'] in self.non_recyclable:
                    self.nonrecyclable_quantity += int(row['Quantity'])
                elif row['Type'] in self.sometimes_recyclable:
                    self.sometimesrecyclable_quantity += int(row['Quantity'])
        self.calculate_total_records()


class DublinGenData(GenPlasticWasteData):
    def __init__(self, population):
        super().__init__("Dublin", population)

    def generate_data(self):
        super().generate_data()

    def generate_records(self):
        """Generates 10,000 random records for Hayward."""
        self.recyclable_quantity = 0
        self.nonrecyclable_quantity = 0
        self.sometimesrecyclable_quantity = 0

        for _ in range(10000):
            plastic_type = random.choice(self.plastic_types)
            quantity = random.randint(1, 100)

            if plastic_type in self.recyclable:
                self.recyclable_quantity += quantity
            elif plastic_type in self.non_recyclable:
                self.nonrecyclable_quantity += quantity
            elif plastic_type in self.sometimes_recyclable:
                self.sometimesrecyclable_quantity += quantity

        self.calculate_total_records()

    def write_csv(self, file_path):
        """Writes the generated data to the CSV file for Hayward."""
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["City", "Population", "Recyclable Quantity", "Non-Recyclable Quantity",
                             "Sometimes Recyclable Quantity", "Date Received", "Date Processed"])
            writer.writerow([self.city, self.population, self.recyclable_quantity,
                             self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                             self.date_received, self.date_processed])

    def read_csv(self, file_path):
        """Reads the CSV file and populates the Hayward data."""
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Type'] in self.recyclable:
                    self.recyclable_quantity += int(row['Quantity'])
                elif row['Type'] in self.non_recyclable:
                    self.nonrecyclable_quantity += int(row['Quantity'])
                elif row['Type'] in self.sometimes_recyclable:
                    self.sometimesrecyclable_quantity += int(row['Quantity'])
        self.calculate_total_records()

# PlasticWasteRecyclingDB to manage plastic waste data using hash table
class PlasticWasteRecyclingDB:
    DELETED = object()  # Special marker for deleted entries

    def __init__(self, capacity=100):
        self.capacity = capacity
        self.type = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]
        self.hash_table = [None] * self.capacity
        self.cities = ["Fremont", "Hayward", "Oakland", "Pleasanton", "Dublin"]

    def hash_function(self, city):
        """A simple hash function to map city to an index."""
        return sum(ord(c) for c in city) % self.capacity

    def insert(self, city, value):
        """Insert the city and its data into the hash table."""
        index = self.hash_function(city)
        if self.hash_table[index] is not None and self.hash_table[index] is not self.DELETED:
            raise ValueError(f"City {city} already exists in the database.")
        self.hash_table[index] = value

    def delete(self, city):
        """Delete the data of a city from the hash table, marking it as 'deleted'."""
        index = self.hash_function(city)
        if self.hash_table[index] is not None and self.hash_table[index] is not self.DELETED:
            self.hash_table[index] = self.DELETED
        else:
            print(f"City {city} not found or already deleted.")

    def search(self, city):
        """Search for a city's data."""
        index = self.hash_function(city)
        data = self.hash_table[index]
        if data is self.DELETED:
            print(f"City {city} has been deleted.")
            return None
        return data

    def classify_recyclable(self, plastic_type):
        """Classify the plastic type as recyclable."""
        if plastic_type in self.type:
            return plastic_type in GenPlasticWasteData.recyclable
        return False

    def classify_nonrecyclable(self, plastic_type):
        """Classify the plastic type as non-recyclable."""
        if plastic_type in self.type:
            return plastic_type in GenPlasticWasteData.non_recyclable
        return False

    def classify_sometimesrecyclable(self, plastic_type):
        """Classify the plastic type as sometimes recyclable."""
        if plastic_type in self.type:
            return plastic_type in GenPlasticWasteData.sometimes_recyclable
        return False

    def read_city_recycle_data(self, city):
        """Reads the recycling data for a given city."""
        city_data = self.search(city)
        if city_data:
            print(f"City: {city_data.city}")
            print(f"Recyclable Quantity: {city_data.get_city_recyclable_data()}")
            print(f"Non-Recyclable Quantity: {city_data.get_city_non_recyclable_data()}")
            print(f"Sometimes Recyclable Quantity: {city_data.get_city_sometimes_recyclable_data()}")
            print(f"Date Received: {city_data.get_date_received()}")
            print(f"Date Processed: {city_data.get_date_processed()}")
        else:
            print(f"No data found for {city}.")

    def compare_efficiency(self, city_data):
        """
        Compares the recycling efficiency of different cities.
        Efficiency is calculated as the ratio of recyclable quantity to total quantity (recyclable + non-recyclable + sometimes recyclable).
        """
        efficiency_data = {}

        for city in city_data:
            total_waste = city.recyclable_quantity + city.nonrecyclable_quantity + city.sometimesrecyclable_quantity
            if total_waste == 0:
                efficiency = 0
            else:
                efficiency = city.recyclable_quantity / total_waste  # Recycling efficiency ratio

            efficiency_data[city.city] = efficiency

        # Sort cities based on efficiency in descending order
        sorted_efficiency = sorted(efficiency_data.items(), key=lambda item: item[1], reverse=True)

        print("City Recycling Efficiency Comparison:")
        for city, efficiency in sorted_efficiency:
            print(f"{city}: {efficiency:.2f}")

        # Plot bar chart for recycling efficiency
        self.plot_efficiency_bar_chart(efficiency_data)

        # Plot pie chart for recycling efficiency distribution
        self.plot_efficiency_pie_chart(efficiency_data)

    # Function to create the UI
    def plot_efficiency_bar_chart(self, efficiency_data):
        """
        Displays a bar chart of recycling efficiency by city.
        """
        cities = list(efficiency_data.keys())
        efficiencies = list(efficiency_data.values())

        plt.bar(cities, efficiencies, color='skyblue')
        plt.xlabel('Cities')
        plt.ylabel('Recycling Efficiency')
        plt.title('Recycling Efficiency by City')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_efficiency_pie_chart(self, efficiency_data):
        """
        Displays a pie chart for recycling efficiency distribution.
        """
        # Pie chart for recycling efficiency
        labels = list(efficiency_data.keys())
        sizes = list(efficiency_data.values())
        colors = plt.cm.Paired(range(len(efficiency_data)))  # Using a color palette

        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.title('Recycling Efficiency Distribution by City')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()


# Binary Search Tree Node for storing plastic data
class BSTNode:
    def __init__(self, plastic_type, city_name, recyclable_quantity, non_recyclable_quantity, sometimes_recyclable_quantity, date_received, date_processed):
        self.plastic_type = plastic_type
        self.city_name = city_name
        self.recyclable_quantity = recyclable_quantity
        self.non_recyclable_quantity = non_recyclable_quantity
        self.sometimes_recyclable_quantity = sometimes_recyclable_quantity
        self.date_received = date_received
        self.date_processed = date_processed
        self.left = None
        self.right = None


# Binary Search Tree (BST) for storing plastic waste data
class PlasticWasteBST:
    def __init__(self):
        self.root = None

    def insert(self, plastic_type, city_name, recyclable_quantity, non_recyclable_quantity, sometimes_recyclable_quantity, date_received, date_processed):
        """Insert a new record into the BST."""
        new_node = BSTNode(plastic_type, city_name, recyclable_quantity, non_recyclable_quantity, sometimes_recyclable_quantity, date_received, date_processed)
        if self.root is None:
            self.root = new_node
        else:
            self._insert(self.root, new_node)

    def _insert(self, node, new_node):
        """Helper method for recursive insertion."""
        if new_node.plastic_type < node.plastic_type:
            if node.left is None:
                node.left = new_node
            else:
                self._insert(node.left, new_node)
        else:
            if node.right is None:
                node.right = new_node
            else:
                self._insert(node.right, new_node)

    def search(self, plastic_type):
        """Search for a node by plastic type."""
        return self._search(self.root, plastic_type)

    def _search(self, node, plastic_type):
        """Helper method for recursive search."""
        if node is None or node.plastic_type == plastic_type:
            return node
        elif plastic_type < node.plastic_type:
            return self._search(node.left, plastic_type)
        else:
            return self._search(node.right, plastic_type)


# Visualizing Plastic Waste Data
class VisualizePlasticWaste:
    def __init__(self):
        self.bst = PlasticWasteBST()

    def add_data(self, plastic_type, city_name, recyclable_quantity, non_recyclable_quantity, sometimes_recyclable_quantity, date_received, date_processed):
        """Add data to the BST."""
        self.bst.insert(plastic_type, city_name, recyclable_quantity, non_recyclable_quantity, sometimes_recyclable_quantity, date_received, date_processed)

    def display_bar_graph(self):
        """Display bar graph of recyclable, non-recyclable, and sometimes recyclable quantities."""
        cities = []
        recyclable_quantities = []
        non_recyclable_quantities = []
        sometimes_recyclable_quantities = []

        def traverse_and_collect_data(node):
            if node:
                cities.append(node.city_name)
                recyclable_quantities.append(node.recyclable_quantity)
                non_recyclable_quantities.append(node.non_recyclable_quantity)
                sometimes_recyclable_quantities.append(node.sometimes_recyclable_quantity)
                traverse_and_collect_data(node.left)
                traverse_and_collect_data(node.right)

        traverse_and_collect_data(self.bst.root)

        # Plot bar graph
        x = np.arange(len(cities))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.bar(x - width, recyclable_quantities, width, label="Recyclable", color="g")
        ax.bar(x, non_recyclable_quantities, width, label="Non-Recyclable", color="r")
        ax.bar(x + width, sometimes_recyclable_quantities, width, label="Sometimes Recyclable", color="b")

        ax.set_xlabel("Cities")
        ax.set_ylabel("Quantities")
        ax.set_title("Plastic Waste Quantities by City")
        ax.set_xticks(x)
        ax.set_xticklabels(cities)
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def display_stacked_bar_graph(self):
        """Display a stacked bar graph for recyclable, non-recyclable, and sometimes recyclable plastics."""
        cities = []
        recyclable_quantities = []
        non_recyclable_quantities = []
        sometimes_recyclable_quantities = []

        def traverse_and_collect_data(node):
            if node:
                cities.append(node.city_name)
                recyclable_quantities.append(node.recyclable_quantity)
                non_recyclable_quantities.append(node.non_recyclable_quantity)
                sometimes_recyclable_quantities.append(node.sometimes_recyclable_quantity)
                traverse_and_collect_data(node.left)
                traverse_and_collect_data(node.right)

        traverse_and_collect_data(self.bst.root)

        # Plot stacked bar graph
        data = np.array([recyclable_quantities, non_recyclable_quantities, sometimes_recyclable_quantities]).T
        fig, ax = plt.subplots(figsize=(10, 7))

        ax.bar(cities, data[:, 0], label="Recyclable", color="g")
        ax.bar(cities, data[:, 1], bottom=data[:, 0], label="Non-Recyclable", color="r")
        ax.bar(cities, data[:, 2], bottom=data[:, 0] + data[:, 1], label="Sometimes Recyclable", color="b")

        ax.set_xlabel("Cities")
        ax.set_ylabel("Quantities")
        ax.set_title("Stacked Plastic Waste Quantities by City")
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def display_pie_chart(self):
        """Display a pie chart for the types of plastics."""
        recyclable = 0
        non_recyclable = 0
        sometimes_recyclable = 0

        def traverse_and_collect_data(node):
            nonlocal recyclable, non_recyclable, sometimes_recyclable
            if node:
                recyclable += node.recyclable_quantity
                non_recyclable += node.non_recyclable_quantity
                sometimes_recyclable += node.sometimes_recyclable_quantity
                traverse_and_collect_data(node.left)
                traverse_and_collect_data(node.right)

        traverse_and_collect_data(self.bst.root)

        # Pie chart for types of plastics
        labels = ["Recyclable", "Non-Recyclable", "Sometimes Recyclable"]
        sizes = [recyclable, non_recyclable, sometimes_recyclable]
        colors = ["g", "r", "b"]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.title("Distribution of Plastic Waste Types")
        plt.show()

    def display_line_graph(self):
        """Display a line graph showing plastic waste quantities over time for each city."""
        dates = []
        city_data = {}

        def traverse_and_collect_data(node):
            if node:
                if node.city_name not in city_data:
                    city_data[node.city_name] = {
                        "dates": [],
                        "recyclable_quantities": [],
                        "non_recyclable_quantities": [],
                        "sometimes_recyclable_quantities": [],
                    }
                city_data[node.city_name]["dates"].append(node.date_received)
                city_data[node.city_name]["recyclable_quantities"].append(node.recyclable_quantity)
                city_data[node.city_name]["non_recyclable_quantities"].append(node.non_recyclable_quantity)
                city_data[node.city_name]["sometimes_recyclable_quantities"].append(node.sometimes_recyclable_quantity)

                traverse_and_collect_data(node.left)
                traverse_and_collect_data(node.right)

        traverse_and_collect_data(self.bst.root)

        # Plot line graph
        fig, ax = plt.subplots(figsize=(10, 7))

        for city, data in city_data.items():
            ax.plot(data["dates"], data["recyclable_quantities"], label=f"{city} - Recyclable", color="g")
            ax.plot(data["dates"], data["non_recyclable_quantities"], label=f"{city} - Non-Recyclable", color="r")
            ax.plot(data["dates"], data["sometimes_recyclable_quantities"], label=f"{city} - Sometimes Recyclable", color="b")

        ax.set_xlabel("Date")
        ax.set_ylabel("Quantities")
        ax.set_title("Plastic Waste Quantities Over Time by City")
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def display_heat_map(self):
        """Display a heat map of recyclable and non-recyclable plastics for cities."""
        cities = []
        recyclable_quantities = []
        non_recyclable_quantities = []

        def traverse_and_collect_data(node):
            if node:
                cities.append(node.city_name)
                recyclable_quantities.append(node.recyclable_quantity)
                non_recyclable_quantities.append(node.non_recyclable_quantity)
                traverse_and_collect_data(node.left)
                traverse_and_collect_data(node.right)

        traverse_and_collect_data(self.bst.root)

        # Create a DataFrame for heatmap
        data = pd.DataFrame({
            "Cities": cities,
            "Recyclable": recyclable_quantities,
            "Non-Recyclable": non_recyclable_quantities
        })

        # Create a pivoted DataFrame for the heatmap
        data_pivot = data.pivot(index="Cities", columns="Recyclable", values="Non-Recyclable")

        # Plot heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(data_pivot, annot=True, cmap="coolwarm", fmt=".2f")  # Changed format specifier to .2f
        plt.title("Heatmap of Plastic Waste Quantities by City")
        plt.tight_layout()
        plt.show()

def create_ui():
    print("In create_ui function")
    root = tk.Tk()
    root.title("Plastic Waste Recycling Efficiency Comparison")

    # List of available cities
    cities = ["Fremont", "Hayward", "Oakland", "Pleasanton", "Dublin"]

    # Initialize variables for user selections
    selected_cities = []

    # Add checkboxes for city selection
    def on_city_select():
        print("In on_city_select function")
        selected_cities.clear()
        for i, var in enumerate(city_vars):
            if var.get() == 1:
                selected_cities.append(cities[i])

    # Function to generate and compare efficiency when the button is clicked
    def compare_efficiency():
        print("In compare_efficiency function")
        if len(selected_cities) < 2:
            messagebox.showerror("Error", "Please select at least two cities for comparison.")
            return

        # Generate data for selected cities
        city_data = []
        for city in selected_cities:
            if city == "Fremont":
                data = fremont_data
            elif city == "Hayward":
                data = hayward_data
            elif city == "Oakland":
                data = oakland_data
            elif city == "Pleasanton":
                data = pleasanton_data
            elif city == "Dublin":
                data = dublin_data

            city_data.append(data)

        # Initialize the database and insert the data
        db = PlasticWasteRecyclingDB(10)
        for city_data_item in city_data:
            db.insert(city_data_item.city, city_data_item)

        # Compare efficiency
        db.compare_efficiency(city_data)

    # Create checkboxes for city selection
    city_vars = []
    for city in cities:
        var = tk.IntVar()
        city_vars.append(var)
        cb = tk.Checkbutton(root, text=city, variable=var, onvalue=1, offvalue=0, command=on_city_select)
        cb.pack()

    # Compare button
    compare_button = tk.Button(root, text="Compare Efficiency", command=compare_efficiency)
    compare_button.pack(pady=10)

    root.mainloop()


# Example Usage
# Creating instances for each city
fremont_data = FremontGenData(population=240000)
hayward_data = HaywardGenData(population=160000)
oakland_data = OaklandGenData(population=420000)
pleasanton_data = PleasantonGenData(population=78000)
dublin_data = DublinGenData(population=60000)

# Generating 10,000 records for each city
fremont_data.generate_records()
hayward_data.generate_records()
oakland_data.generate_records()
pleasanton_data.generate_records()
dublin_data.generate_records()

# Writing the generated data to CSV files
fremont_data.write_csv("fremont_plastic_waste_data.csv")
hayward_data.write_csv("hayward_plastic_waste_data.csv")
oakland_data.write_csv("oakland_plastic_waste_data.csv")
pleasanton_data.write_csv("pleasanton_plastic_waste_data.csv")
dublin_data.write_csv("dublin_plastic_waste_data.csv")

# Create the PlasticWasteRecyclingDB and insert data for Fremont
db = PlasticWasteRecyclingDB(capacity=10)
db.insert("Fremont", fremont_data)

# Read the data back for Fremont
db.read_city_recycle_data("Fremont")
db.read_city_recycle_data("Hayward")
db.read_city_recycle_data("Oakland")
db.read_city_recycle_data("Pleasanton")
db.read_city_recycle_data("Dublin")

# db.compare_efficiency([fremont_data, hayward_data, oakland_data, pleasanton_data, dublin_data])

# Delete the Fremont entry
db.delete("Fremont")

# Try searching after deletion
db.read_city_recycle_data("Fremont")

# Classify plastic types
print(db.classify_recyclable("PET"))  # Should return True
print(db.classify_nonrecyclable("PVC"))  # Should return True
print(db.classify_sometimesrecyclable("LDPE"))  # Should return True

# Example usage
visualizer = VisualizePlasticWaste()

# Add some sample data
visualizer.add_data("PET", "Fremont", 500, 200, 100, datetime(2025, 4, 1), datetime(2025, 4, 3))
visualizer.add_data("HDPE", "Fremont", 600, 150, 200, datetime(2025, 4, 2), datetime(2025, 4, 4))
visualizer.add_data("PVC", "Hayward", 300, 400, 150, datetime(2025, 4, 3), datetime(2025, 4, 5))

# Display various visualizations
visualizer.display_bar_graph()
visualizer.display_stacked_bar_graph()
visualizer.display_pie_chart()
visualizer.display_line_graph()
visualizer.display_heat_map()

create_ui()