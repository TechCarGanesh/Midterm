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
import os

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
        # Abstract method.
        raise NotImplementedError("This method must be implemented in a subclass.")

    def write_csv(self, file_path):
        # Abstract method.
        raise NotImplementedError("This method must be implemented in a subclass.")

    def read_csv(self, file_path):
        # Abstract method.
        raise NotImplementedError("This method must be implemented in a subclass.")

    def set_processed_date(self):
        # Sets the date when the data was processed.
        self.date_processed = date.today()

    def calculate_total_records(self):
        # Calculates the total number of records based on the quantities.
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
        self.monthly_data = [] * 12
        super().__init__("Fremont", population)

    def generate_data(self, filename):
        # Generates 10,000 records of plastic waste data and writes them to a CSV file.
        # Each record will be in the format: "type, quantity" pairs.
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for _ in range(10000):
                record = []
                for plastic_type in self.plastic_types:
                    quantity = random.randint(50, 500)  # Generate a random quantity for each plastic type
                    record.append(plastic_type)
                    record.append(quantity)
                    if plastic_type in self.recyclable:
                        self.recyclable_quantity += quantity
                    elif plastic_type in self.non_recyclable:
                        self.nonrecyclable_quantity += quantity
                    elif plastic_type in self.sometimes_recyclable:
                        self.sometimesrecyclable_quantity += quantity
                self.calculate_total_records()
                writer.writerow(record)
            self.monthly_data.append(
                [self.recyclable_quantity, self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                 self.date_received, self.date_processed])

    def calculate_total_records(self):
        # Calculates the total number of records based on the quantities.
        self.total_records = (self.recyclable_quantity +
                              self.nonrecyclable_quantity +
                              self.sometimesrecyclable_quantity)


    def read_csv(self, filename):
        # Reads the generated CSV file and returns the data in a list of dictionaries
        # where each dictionary contains the plastic type and quantity.
        data = []
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                record = {}
                for i in range(0, len(row), 2):  # Pairing the plastic type and quantity
                    plastic_type = row[i]
                    quantity = int(row[i + 1])
                    record[plastic_type] = quantity
                    if plastic_type in self.recyclable:
                        self.recyclable_quantity += quantity
                    elif plastic_type in self.non_recyclable:
                        self.nonrecyclable_quantity += quantity
                    elif plastic_type in self.sometimes_recyclable:
                        self.sometimesrecyclable_quantity += quantity
                self.calculate_total_records()
                data.append(record)
            self.monthly_data.append(
                [self.recyclable_quantity, self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                 self.date_received, self.date_processed])
        return data


    def get_monthly_data(self, month):
        return self.monthly_data[month]


# Subclass for Hayward
class HaywardGenData(GenPlasticWasteData):
    def __init__(self, population):
        self.monthly_data = [] * 12
        super().__init__("Hayward", population)

    def generate_data(self, filename):
        # Generates 10,000 records of plastic waste data and writes them to a CSV file.
        # Each record will be in the format: "type, quantity" pairs.
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for _ in range(10000):
                record = []
                for plastic_type in self.plastic_types:
                    quantity = random.randint(50, 500)  # Generate a random quantity for each plastic type
                    record.append(plastic_type)
                    record.append(quantity)
                    if plastic_type in self.recyclable:
                        self.recyclable_quantity += quantity
                    elif plastic_type in self.non_recyclable:
                        self.nonrecyclable_quantity += quantity
                    elif plastic_type in self.sometimes_recyclable:
                        self.sometimesrecyclable_quantity += quantity
                self.calculate_total_records()
                writer.writerow(record)
            self.monthly_data.append(
                [self.recyclable_quantity, self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                 self.date_received, self.date_processed])

    def read_csv(self, filename):
        # Reads the generated CSV file and returns the data in a list of dictionaries
        # where each dictionary contains the plastic type and quantity.
        data = []
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                record = {}
                for i in range(0, len(row), 2):  # Pairing the plastic type and quantity
                    plastic_type = row[i]
                    quantity = int(row[i + 1])
                    record[plastic_type] = quantity
                    if plastic_type in self.recyclable:
                        self.recyclable_quantity += quantity
                    elif plastic_type in self.non_recyclable:
                        self.nonrecyclable_quantity += quantity
                    elif plastic_type in self.sometimes_recyclable:
                        self.sometimesrecyclable_quantity += quantity
                data.append(record)
            self.monthly_data.append(
                [self.recyclable_quantity, self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                 self.date_received, self.date_processed])
        return data

    def get_monthly_data(self, month):
        return self.monthly_data[month]

# Repeat similar classes for other cities (Oakland, Pleasanton, Dublin)
class OaklandGenData(GenPlasticWasteData):
    def __init__(self, population):
        self.monthly_data = [] * 12
        super().__init__("Oakland", population)

    def generate_data(self, filename):
        # Generates 10,000 records of plastic waste data and writes them to a CSV file.
        # Each record will be in the format: "type, quantity" pairs.
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for _ in range(10000):
                record = []
                for plastic_type in self.plastic_types:
                    quantity = random.randint(50, 500)  # Generate a random quantity for each plastic type
                    record.append(plastic_type)
                    record.append(quantity)
                    if plastic_type in self.recyclable:
                        self.recyclable_quantity += quantity
                    elif plastic_type in self.non_recyclable:
                        self.nonrecyclable_quantity += quantity
                    elif plastic_type in self.sometimes_recyclable:
                        self.sometimesrecyclable_quantity += quantity
                self.calculate_total_records()
                writer.writerow(record)
            self.monthly_data.append(
                [self.recyclable_quantity, self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                 self.date_received, self.date_processed])

    def read_csv(self, filename):
        # Reads the generated CSV file and returns the data in a list of dictionaries
        # where each dictionary contains the plastic type and quantity.
        data = []
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                record = {}
                for i in range(0, len(row), 2):  # Pairing the plastic type and quantity
                    plastic_type = row[i]
                    quantity = int(row[i + 1])
                    record[plastic_type] = quantity
                    if plastic_type in self.recyclable:
                        self.recyclable_quantity += quantity
                    elif plastic_type in self.non_recyclable:
                        self.nonrecyclable_quantity += quantity
                    elif plastic_type in self.sometimes_recyclable:
                        self.sometimesrecyclable_quantity += quantity
                data.append(record)
            self.monthly_data.append(
                [self.recyclable_quantity, self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                 self.date_received, self.date_processed])
        return data


    def get_monthly_data(self, month):
        return self.monthly_data[month]

class PleasantonGenData(GenPlasticWasteData):
    def __init__(self, population):
        self.monthly_data = [] * 12
        super().__init__("Pleasanton", population)

    def generate_data(self, filename):
        # Generates 10,000 records of plastic waste data and writes them to a CSV file.
        # Each record will be in the format: "type, quantity" pairs.
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for _ in range(10000):
                record = []
                for plastic_type in self.plastic_types:
                    quantity = random.randint(50, 500)  # Generate a random quantity for each plastic type
                    record.append(plastic_type)
                    record.append(quantity)
                    if plastic_type in self.recyclable:
                        self.recyclable_quantity += quantity
                    elif plastic_type in self.non_recyclable:
                        self.nonrecyclable_quantity += quantity
                    elif plastic_type in self.sometimes_recyclable:
                        self.sometimesrecyclable_quantity += quantity
                self.calculate_total_records()
                writer.writerow(record)
            self.monthly_data.append(
                [self.recyclable_quantity, self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                 self.date_received, self.date_processed])

    def read_csv(self, filename):
        # Reads the generated CSV file and returns the data in a list of dictionaries
        # where each dictionary contains the plastic type and quantity.
        data = []
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                record = {}
                for i in range(0, len(row), 2):  # Pairing the plastic type and quantity
                    plastic_type = row[i]
                    quantity = int(row[i + 1])
                    record[plastic_type] = quantity
                    if plastic_type in self.recyclable:
                        self.recyclable_quantity += quantity
                    elif plastic_type in self.non_recyclable:
                        self.nonrecyclable_quantity += quantity
                    elif plastic_type in self.sometimes_recyclable:
                        self.sometimesrecyclable_quantity += quantity
                data.append(record)
            self.monthly_data.append(
                [self.recyclable_quantity, self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                 self.date_received, self.date_processed])
        return data

    def get_monthly_data(self, month):
        return self.monthly_data[month]


class DublinGenData(GenPlasticWasteData):
    def __init__(self, population):
        self.monthly_data = [] * 12
        super().__init__("Dublin", population)

    def generate_data(self, filename):
        # Generates 10,000 records of plastic waste data and writes them to a CSV file.
        # Each record will be in the format: "type, quantity" pairs.
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for _ in range(10000):
                record = []
                for plastic_type in self.plastic_types:
                    quantity = random.randint(50, 500)  # Generate a random quantity for each plastic type
                    record.append(plastic_type)
                    record.append(quantity)
                    if plastic_type in self.recyclable:
                        self.recyclable_quantity += quantity
                    elif plastic_type in self.non_recyclable:
                        self.nonrecyclable_quantity += quantity
                    elif plastic_type in self.sometimes_recyclable:
                        self.sometimesrecyclable_quantity += quantity
                self.calculate_total_records()
                writer.writerow(record)
            self.monthly_data.append(
                [self.recyclable_quantity, self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                 self.date_received, self.date_processed])

    def read_csv(self, filename):
        # Reads the generated CSV file and returns the data in a list of dictionaries
        # where each dictionary contains the plastic type and quantity.
        data = []
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                record = {}
                for i in range(0, len(row), 2):  # Pairing the plastic type and quantity
                    plastic_type = row[i]
                    quantity = int(row[i + 1])
                    record[plastic_type] = quantity
                    if plastic_type in self.recyclable:
                        self.recyclable_quantity += quantity
                    elif plastic_type in self.non_recyclable:
                        self.nonrecyclable_quantity += quantity
                    elif plastic_type in self.sometimes_recyclable:
                        self.sometimesrecyclable_quantity += quantity
                data.append(record)
            self.monthly_data.append(
                [self.recyclable_quantity, self.nonrecyclable_quantity, self.sometimesrecyclable_quantity,
                 self.date_received, self.date_processed])
        return data


    def get_monthly_data(self, month):
        return self.monthly_data[month]


# PlasticWasteRecyclingDB to manage plastic waste data using hash table
class PlasticWasteRecyclingDB:
    def __init__(self, capacity=101):
        self.capacity = capacity
        self.type = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]
        self.hash_table = [None] * self.capacity
        self.cities = ["Fremont", "Hayward", "Oakland", "Pleasanton", "Dublin"]
        self.deleted_marker = "DELETED"
        self.num_collisions = 0
        self.curr_size = 0
        self.load_average = 0

    def hash_function(self, city):
        # A simple hash function to map city to an index.
        return sum(ord(c) for c in city) % self.capacity


    def insert_hash(self, key, value):
        index = self.hash_function(key)
        original_index = index

        while self.hash_table[index] is not None:
            if self.hash_table[index][0] == key:
                # Update existing value
                self.hash_table[index] = (key, value)
                return
            index = (index + 1) % self.capacity
            self.num_collisions += 1
            if index == original_index:
                raise Exception("Hash table is full")
            elif self.hash_table[index] == None or self.hash_table[index] == self.deleted_marker:
                self.hash_table[index] = (key, value)
                self.curr_size += 1
                self.load_average = (self.curr_size/self.capacity)
                return

        # No collisions Insert new key-value pair
        self.hash_table[index] = (key, value)
        self.curr_size += 1
        self.load_average = (self.curr_size/self.capacity)
        return

    def delete_hash(self, key):
        # Delete the data of a city from the hash table, marking it as 'deleted'
        index = self.hash_function(key)
        original_index = index
        while self.hash_table[index] is not None:
            if self.hash_table[index][0] == key:
                self.hash_table[index] = self.deleted_marker
                return
            index = (index + 1) % self.capacity
            if index == original_index:
                raise Exception(f"{key} not found")
        if self.hash_table[index] == self.deleted_marker:
            print(f"City {key} not found or already deleted.")
        return

    def search_hash(self, key):
        # Generate hash index using city + month
        # For example "Fremont" + "January"
        index = self.hash_function(key)
        original_index = index
        while self.hash_table[index] is not None:
            if self.hash_table[index][0] == key:
                return self.hash_table[index]
            index = (index + 1) % self.capacity
            if index == original_index:
                raise Exception(f"{key} not found")
        value = self.hash_table[index]
        if self.hash_table[index] is self.deleted_marker:
            print(f"City {key} has been deleted.")
            return None
        return value

    def classify_recyclable(self, plastic_type):
        # Classify the plastic type as recyclable.
        if plastic_type in self.type:
            return plastic_type in GenPlasticWasteData.recyclable
        return False

    def classify_nonrecyclable(self, plastic_type):
        # Classify the plastic type as non-recyclable.
        if plastic_type in self.type:
            return plastic_type in GenPlasticWasteData.non_recyclable
        return False

    def classify_sometimesrecyclable(self, plastic_type):
        # Classify the plastic type as sometimes recyclable.
        if plastic_type in self.type:
            return plastic_type in GenPlasticWasteData.sometimes_recyclable
        return False

    def convert_month_to_index(self, month_list):
        # List of months in order
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]

        # Create a mapping of month name to index
        month_to_index = {month: index for index, month in enumerate(months)}

        # Convert the input list to indices
        return [month_to_index[month] for month in month_list]

    def compare_efficiency(self, city_data, selected_months):
        # Compares the recycling efficiency of different cities.
        # Efficiency is calculated as the ratio of recyclable quantity to total quantity
        # (recyclable + non-recyclable + sometimes recyclable) * population.
        efficiency_data = {}
        mlist = self.convert_month_to_index(selected_months)

        for month in mlist:
            for city in city_data:
                data = city.get_monthly_data(month)
                print("month ", month, "City ", city.city, "Data ", data)
                total_waste = (data[0] + data[1] + data[2]) * city.population
                recyclable_waste = data[0]
                if total_waste == 0:
                    efficiency = 0
                    efficiency_data[city.city] = efficiency
                else:
                    efficiency = (recyclable_waste / total_waste)  # Recycling efficiency ratio
                    efficiency_data[city.city] = efficiency

        # Sort cities based on efficiency in descending order
        sorted_efficiency = sorted(efficiency_data.items(), key=lambda item: item[1], reverse=True)

        # Plot bar chart for recycling efficiency
        print("Efficiency Data ", efficiency_data)
        self.plot_efficiency_bar_chart(efficiency_data)

        # Plot pie chart for recycling efficiency distribution
        self.plot_efficiency_pie_chart(efficiency_data)

    # Function to create the UI
    def plot_efficiency_bar_chart(self, efficiency_data):
        # Displays a bar chart of recycling efficiency by city.
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
        # Displays a pie chart for recycling efficiency distribution.
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

    def insert_bst(self, plastic_type, city_name, recyclable_quantity, non_recyclable_quantity,
                   sometimes_recyclable_quantity, date_received, date_processed):
        # Insert a new record into the BST.
        new_node = BSTNode(plastic_type, city_name, recyclable_quantity, non_recyclable_quantity,
                           sometimes_recyclable_quantity, date_received, date_processed)
        if self.root is None:
            self.root = new_node
        else:
            self.insert_bst_recursive(self.root, new_node)

    def insert_bst_recursive(self, node, new_node):
        # Helper method for recursive insertion.
        if new_node.plastic_type < node.plastic_type:
            if node.left is None:
                node.left = new_node
            else:
                self.insert_bst_recursive(node.left, new_node)
        else:
            if node.right is None:
                node.right = new_node
            else:
                self.insert_bst_recursive(node.right, new_node)

    def search_bst(self, plastic_type):
        # Search for a node by plastic type.
        return self._search(self.root, plastic_type)

    def search_bst_recursive(self, node, plastic_type):
        # Helper method for recursive search.
        if node is None or node.plastic_type == plastic_type:
            return node
        elif plastic_type < node.plastic_type:
            return self.search_bst_recursive(node.left, plastic_type)
        else:
            return self.search_bst_recursive(node.right, plastic_type)


# Visualizing Plastic Waste Data
class VisualizePlasticWaste:
    def __init__(self):
        self.bst = PlasticWasteBST()

    def add_data(self, plastic_type, city_name, recyclable_quantity, non_recyclable_quantity, sometimes_recyclable_quantity, date_received, date_processed):
        # Add data to the BST.
        self.bst.insert_bst(plastic_type, city_name, recyclable_quantity, non_recyclable_quantity, sometimes_recyclable_quantity, date_received, date_processed)

    def display_bar_graph(self):
        # Display bar graph of recyclable, non-recyclable, and sometimes recyclable quantities.
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
        # Display a stacked bar graph for recyclable, non-recyclable, and sometimes recyclable plastics.
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
        # Display a pie chart for the types of plastics.
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
        # Display a line graph showing plastic waste quantities over time for each city.
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
        # Display a heat map of recyclable and non-recyclable plastics for cities.
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
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]

    # Initialize variables for user selections
    selected_cities = []
    selected_months = []
    selected_rate = 0

    # Add a frame to group the checkboxes
    city_frame = tk.Frame(root)
    city_frame.pack(pady=20)  # Add some vertical space between the frame and other widgets

    # Add a frame to group the month checkboxes
    month_frame = tk.Frame(root)
    month_frame.pack(pady=20)

    # Add checkboxes for city selection
    def on_city_select():
        print("In on_city_select function")
        selected_cities.clear()
        for i, var in enumerate(city_vars):
            if var.get() == 1:
                selected_cities.append(cities[i])

    def on_months_select():
        print("In on_month_select function")
        selected_months.clear()
        for i, var in enumerate(month_vars):
            if var.get() == 1:
                selected_months.append(months[i])


    # Function to generate and compare efficiency when the button is clicked
    def compare_efficiency():
        print("In compare_efficiency function")
        # Initialize the database and insert the data
        percentage = 0.0
        db = PlasticWasteRecyclingDB(101)
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
            for month in months:
                filename = f"{city}{month}.csv"
                data_list = data.read_csv(filename)  # Read the generated data
                # Assuming we want to sum the quantities for the city data
                total_recyclable = sum(record.get("PET", 0) + record.get("HDPE", 0) for record in data_list)
                total_non_recyclable = sum(record.get("PVC", 0) for record in data_list)
                total_sometimes_recyclable = sum(
                    record.get("LDPE", 0) + record.get("PP", 0) + record.get("PS", 0) + record.get("Other", 0) for record in
                    data_list)
            data.recyclable_quantity = total_recyclable
            data.nonrecyclable_quantity = total_non_recyclable
            data.sometimesrecyclable_quantity = total_sometimes_recyclable
            city_data.append(data)
        #
        # for city_data_item in city_data:
        #     db.insert_hash(city_data_item.city, city_data_item.monthly_data)

        # Compare efficiency
        db.compare_efficiency(city_data, selected_months)

    # Create checkboxes for city selection inside the frame
    city_vars = []
    for city in cities:
        var = tk.IntVar()
        city_vars.append(var)
        cb = tk.Checkbutton(city_frame, text=city, variable=var, onvalue=1, offvalue=0, command=on_city_select)
        cb.pack(side='left', padx=10)  # Place checkboxes horizontally with padding between them

    # Create checkboxes for city selection
    month_vars = []
    for month in months:
        var = tk.IntVar()
        month_vars.append(var)
        cb = tk.Checkbutton(month_frame, text=month, variable=var, onvalue=1, offvalue=0, command=on_months_select)
        cb.pack(side='left', padx=10)

    # Compare button
    compare_button = tk.Button(root, text="Compare Efficiency", command=compare_efficiency)
    compare_button.pack(pady=10)

    root.mainloop()


# Creating instances for each city
# Global data
fremont_data = FremontGenData(population=240000)
hayward_data = HaywardGenData(population=160000)
oakland_data = OaklandGenData(population=420000)
pleasanton_data = PleasantonGenData(population=78000)
dublin_data = DublinGenData(population=60000)

def main():
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
    cities = ["Fremont", "Hayward", "Oakland", "Pleasanton", "Dublin"]
    hkey =""
    # Create the PlasticWasteRecyclingDB and insert data for the cities
    db = PlasticWasteRecyclingDB(capacity=101)

    # Create VisualizePlasticWaste object
    visualizer = VisualizePlasticWaste()

    for month in months:
        for city in cities:
            file_path = city + month + ".csv"
            hkey = city+month
            # Generating 10,000 records for each city
            if city == "Fremont":
                if not os.path.exists(file_path):
                    fremont_data.generate_data(file_path)
                    fremont_data.month = month
                    db.insert_hash(hkey, fremont_data)
                else:
                    fremont_data.read_csv(file_path)
                    fremont_data.month = month
                    db.insert_hash(hkey, fremont_data)
            if city == "Hayward":
                if not os.path.exists(file_path):
                    hayward_data.generate_data(file_path)
                    hayward_data.month = month
                    db.insert_hash(hkey, hayward_data)
                else:
                    hayward_data.read_csv(file_path)
                    hayward_data.month = month
                    db.insert_hash(hkey, hayward_data)

            if city == "Oakland":
                if not os.path.exists(file_path):
                    oakland_data.generate_data(file_path)
                    oakland_data.month = month
                    db.insert_hash(hkey, oakland_data)
                else:
                    oakland_data.read_csv(file_path)
                    oakland_data.month = month
                    db.insert_hash(hkey, oakland_data)
            if city == "Pleasanton":
                if not os.path.exists(file_path):
                    pleasanton_data.generate_data(file_path)
                    pleasanton_data.month = month
                    db.insert_hash(hkey, pleasanton_data)
                else:
                    pleasanton_data.read_csv(file_path)
                    pleasanton_data.month = month
                    db.insert_hash(hkey, pleasanton_data)
            if city == "Dublin":
                if not os.path.exists(file_path):
                    dublin_data.generate_data(file_path)
                    dublin_data.month = month
                    db.insert_hash(hkey, dublin_data)
                else:
                    dublin_data.read_csv(file_path)
                    dublin_data.month = month
                    db.insert_hash(hkey, dublin_data)


    city_data = [fremont_data, hayward_data, oakland_data, pleasanton_data, dublin_data]
    for month in ["January"]:
        for data in city_data:
            visualizer.add_data(data.recyclable[0], data.city,
                                data.recyclable_quantity / data.population,
                                data.nonrecyclable_quantity / data.population,
                                data.sometimesrecyclable_quantity / data.population,
                                datetime(2025, 4, 1), datetime(2025, 4, 3))
    # Display various visualizations
    visualizer.display_bar_graph()
    visualizer.display_stacked_bar_graph()
    visualizer.display_pie_chart()
    # visualizer.display_line_graph()
    visualizer.display_heat_map()

    for month in ["February", "March"]:
        for data in city_data:
            visualizer.add_data(data.recyclable[0], data.city,
                                data.recyclable_quantity / data.population,
                                data.nonrecyclable_quantity / data.population,
                                data.sometimesrecyclable_quantity / data.population,
                                datetime(2025, 4, 1), datetime(2025, 4, 3))

    create_ui()


if __name__ == "__main__":
    main()
