#! /usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import os
import random
import re
import shlex
import string
import sqlite3
import subprocess

from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlite3 import Error

import colorama
import yaml
from colorama import Fore
from datetime import datetime
from kubernetes import client, config, stream


class DataStorageManager:
    """Manages data storage and retrieval using SQLite."""

    def __init__(self, db_name=".sqlite_database.db"):
        """
        Initializes the DataStorageManager with a specific SQLite database file name.
        The database file will be located in the script's directory.

        :param db_name: The database file name. Defaults to 'sqlite_database.db'.
        """
        script_dir_path = os.path.join(os.path.dirname(__file__), db_name)
        fallback_dir_path = os.path.join(os.path.expanduser("~"), ".nsys_k8s", db_name)
        self.db_file = script_dir_path
        self.conn = self.create_connection(fallback_path=fallback_dir_path)
        if self.conn is not None:
            self.create_output_options_table()

    def create_connection(self, fallback_path):
        """
        Attempt to create a database connection to the SQLite database specified by the db_file.
        If the connection cannot be established at the primary location (self.db_file),
        the method tries to connect at a fallback location specified by the fallback_path parameter.

        :param fallback_path: The fallback path for the database file if the initial connection attempt fails.
        :return: The connection object if the connection was successfully established; None otherwise.
        """
        try:
            conn = sqlite3.connect(self.db_file)
            return conn
        except Error as e:
            try:
                # Ensure the fallback directory exists
                os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
                self.db_file = fallback_path
                conn = sqlite3.connect(self.db_file)
                return conn
            except Error as e:
                print(f"Failed to create database at fallback location. Error: {e}")
                return None

    def create_output_options_table(self):
        """
        Create a table for storing output options if it does not already exist.
        """
        try:
            c = self.conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS output_option_storage (
                            id INTEGER PRIMARY KEY,
                            namespace TEXT NOT NULL,
                            pod_name TEXT NOT NULL,
                            container_name TEXT NOT NULL,
                            output_option TEXT NOT NULL
                        );""")
        except Error as e:
            print(e)

    def insert_output_option(self, namespace, pod_name, container_name, output_option):
        """
        Insert output_option into the table.

        :param namespace: The namespace of the data.
        :param pod_name: The pod name.
        :param container_name: The container name.
        :param output_option: The output option to be stored.
        """
        try:
            sql = """INSERT INTO output_option_storage(namespace,pod_name,container_name,output_option)
                    VALUES(?,?,?,?)"""
            cur = self.conn.cursor()
            cur.execute(sql, (namespace, pod_name, container_name, output_option))
            self.conn.commit()
            return cur.lastrowid
        except Error as e:
            print(e)
            return None

    def retrieve_output_options(self, namespace, pod_name, container_name):
        """
        Retrieve output_options from the table for a specific namespace, pod name, and container name.

        :param namespace: The namespace to filter by.
        :param pod_name: The pod name to filter by.
        :param container_name: The container name to filter by.
        :return: A list of output_option values matching the criteria.
        """
        try:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT output_option FROM output_option_storage WHERE namespace=? AND"
                " pod_name=? AND container_name=?",
                (namespace, pod_name, container_name),
            )
            rows = cur.fetchall()
            return [row[0] for row in rows]
        except Error as e:
            print(e)
            return []


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "This script facilitates the management of Nsight Systems, running through"
            " a NVIDIA DevTools Sidecar Injector."
        )
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    parser_nsys = subparsers.add_parser("nsys", help="Execute nsys command.")
    parser_nsys.add_argument(
        "nsys_args",
        nargs=argparse.REMAINDER,
        help="Nsight Systems arguments to execute.",
    )

    parser_copy = subparsers.add_parser(
        "download", help="Download profiling results command."
    )
    parser_copy.add_argument(
        "destination",
        type=str,
        help=(
            "Specifies the local directory to which profiling results will be"
            " downloaded."
        ),
    )
    parser_copy.add_argument(
        "--remove-source", action="store_true", help="Delete results after copying."
    )

    check_parser = subparsers.add_parser(
        "check",
        help="Check a NVIDIA DevTools Sidecar Injector injected into a specific pod.",
    )
    check_parser.add_argument("pod_name", help="Name of the pod to check.")
    check_parser.add_argument(
        "-n",
        "--namespace",
        default="default",
        help='Namespace of the pod. Defaults to "default".',
    )

    parser.add_argument(
        "--field-selector",
        type=str,
        help=(
            "Filter Kubernetes objects based on the value of one or more resource"
            " fields."
            "https://kubernetes.io/docs/concepts/overview/working-with-objects/field-selectors/"
        ),
        required=False,
    )
    return parser.parse_args()


def load_yaml_file(file_path):
    """
    Loads and parses a YAML file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The parsed YAML data.
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)


def get_helm_values(release_name, namespace=None):
    """
    Retrieves merged helm values for a given release.
    If namespace is specified, helm will look for the release in that namespace.

    Args:
        release_name (str): The name of the release.
        namespace (str, optional): The namespace to look for the release. Defaults to None.

    Returns:
        dict: The merged helm values.
    """
    cmd = ["helm", "get", "values", release_name]
    if namespace:
        cmd += ["-n", namespace]
    # Adding '--all' to get all values, including those overridden or merged
    cmd.append("--all")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return yaml.safe_load(result.stdout)


def generate_uid(length=8):
    """
    Generates a random alphanumeric UID.

    Args:
        length (int, optional): The length of the UID. Defaults to 8.

    Returns:
        str: The generated UID.
    """
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def get_profiling_params(namespace, pod_name, container_name, session):
    """
    Fetches all key-value pairs from a pod/container and returns them as a dictionary.

    Args:
        namespace (str): The namespace of the pod.
        pod_name (str): The name of the pod.
        container_name (str): The name of the container.
        session (str): The session ID.

    Returns:
        dict: A dictionary containing the key-value pairs from the pod/container.
    """
    core_v1_api = client.CoreV1Api()
    exec_command = ["cat", f"/tmp/devtool-injection-{session}"]
    resp = stream.stream(
        core_v1_api.connect_get_namespaced_pod_exec,
        pod_name,
        namespace,
        container=container_name,
        command=exec_command,
        stderr=True,
        stdin=False,
        stdout=True,
        tty=False,
    )

    # Process response to extract all key-value pairs into a dictionary
    lines = resp.splitlines()  # Split the response into lines
    kv_pairs = dict(
        line.split("=", 1) for line in lines if "=" in line
    )  # Create a dict from KEY=VALUE pairs
    kv_pairs = {"".join(k.split()): "".join(v.split()) for k, v in kv_pairs.items()}

    return kv_pairs


def generate_timestamp():
    """
    Generates a millisecond timestamp.

    Returns:
        int: The millisecond timestamp.
    """
    return int(datetime.now().timestamp() * 1000)


def fetch_pod_env_vars(namespace, pod_name, container_name):
    """
    Fetches environment variables from a running container in a pod.

    Args:
        namespace (str): The namespace of the pod.
        pod_name (str): The name of the pod.
        container_name (str): The name of the container.

    Returns:
        dict: A dictionary containing the environment variables as key-value pairs.
    """
    core_v1_api = client.CoreV1Api()
    try:
        exec_command = ["env"]
        response = stream.stream(
            core_v1_api.connect_get_namespaced_pod_exec,
            pod_name,
            namespace,
            container=container_name,
            command=exec_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
        )
        env_vars = dict(
            line.split("=", 1) for line in response.splitlines() if "=" in line
        )
        env_vars = {"".join(k.split()): "".join(v.split()) for k, v in env_vars.items()}
        return env_vars
    except client.ApiException as e:
        print(f"An error occurred while fetching environment variables: {e}")
        return {}


def replace_placeholders(
    namespace, pod_name, container_name, env_vars, session, option
):
    """
    Replaces placeholders in the output_option string.

    Args:
        namespace (str): The namespace of the pod.
        pod_name (str): The name of the pod.
        container_name (str): The name of the container.
        env_vars (dict): A dictionary containing the environment variables as key-value pairs.
        session: The session object.
        option (str): The output_option string.

    Returns:
        str: The modified output_option string.
    """
    uid = generate_uid()
    option = option.replace("{UID}", uid)

    profiling_params = get_profiling_params(
        namespace, pod_name, container_name, session
    )
    for key, value in profiling_params.items():
        option = option.replace(f"{{{key}}}", value)

    timestamp = generate_timestamp()
    option = option.replace("{TIMESTAMP}", str(timestamp))

    for key, value in env_vars.items():
        option = option.replace(f"%{{{key}}}", value)

    return option


def run_shell_command(command):
    """
    Executes a shell command and returns its output.

    Args:
        command (str): The shell command to execute.

    Returns:
        str: The output of the shell command.
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e.stderr}")
        return None


def initialize_kubernetes_client():
    """
    Loads the kubeconfig file and initializes the Kubernetes client.
    """
    try:
        config.load_kube_config()  # Load the kubeconfig file
    except Exception as e:
        print(f"Error loading Kubernetes config: {e}")
        exit(1)


def list_namespaces_with_label(label_selector):
    """
    Lists namespaces with a specific label.

    Args:
        label_selector (str): The label selector used to filter namespaces.

    Returns:
        list: A list of namespaces that match the label selector.
    """
    core_v1_api = client.CoreV1Api()
    try:
        return core_v1_api.list_namespace(label_selector=label_selector)
    except client.exceptions.ApiException as e:
        print(f"An error occurred while listing namespaces: {e}")
        exit(1)


def list_pods_with_label(field_selector, label_selector):
    """
    Lists pods across all namespaces with a specific label.

    Args:
        field_selector (str): The field selector used to filter pods.
        label_selector (str): The label selector used to filter pods.

    Returns:
        list: A list of pods that match the field and label selectors.
    """
    core_v1_api = client.CoreV1Api()
    try:
        return core_v1_api.list_pod_for_all_namespaces(
            field_selector=field_selector, label_selector=label_selector
        )
    except client.exceptions.ApiException as e:
        print(f"An error occurred while listing pods: {e}")
        exit(1)


def list_pods_in_namespace(field_selector, namespace):
    """
    Lists all pods within a specific namespace.

    Args:
        field_selector (str): The field selector used to filter pods.
        namespace (str): The namespace to list pods from.

    Returns:
        list: A list of pods within the specified namespace.
    """
    core_v1_api = client.CoreV1Api()
    try:
        return core_v1_api.list_namespaced_pod(namespace, field_selector=field_selector)
    except client.exceptions.ApiException as e:
        print(f"An error occurred while listing pods in namespace {namespace}: {e}")
        exit(1)


def get_unique_kubernetes_pods(
    field_selector, namespace_label_selector, pod_label_selector
):
    """
    Returns a unique list of pods based on namespace labels and pod labels.

    Args:
        field_selector (str): The field selector used to filter pods.
        namespace_label_selector (str): The label selector used to filter namespaces.
        pod_label_selector (str): The label selector used to filter pods.

    Returns:
        list: A list of unique pods based on the specified field and label selectors.
    """
    unique_pods = {}

    # Pods from labeled namespaces
    namespaces = list_namespaces_with_label(namespace_label_selector)
    for ns in namespaces.items:
        pods = list_pods_in_namespace(field_selector, ns.metadata.name)
        for pod in pods.items:
            unique_pods[f"{pod.metadata.namespace}/{pod.metadata.name}"] = pod

    # Labeled pods across all namespaces
    labeled_pods = list_pods_with_label(field_selector, pod_label_selector)
    for pod in labeled_pods.items:
        unique_pods[f"{pod.metadata.namespace}/{pod.metadata.name}"] = pod

    return list(unique_pods.values())


def extract_output_args_from_nsys_args_str(nsys_args_str):
    """
    Extracts the output path from NSYS arguments.

    Args:
        nsys_args_str (str): The NSYS arguments as a string.

    Returns:
        list or None: The output path as a list if found, None otherwise.
    """
    match = re.search(
        r'(?:--output|-o)\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))', nsys_args_str
    )
    if match:
        return ["-o", next(s for s in match.groups() if s)]
    return None


def extract_output_args_from_nsys_args(nsys_args):
    """
    Extracts the output option and its value from NSYS arguments and returns the cleaned arguments
    list without the output option and the extracted output option as a string.

    Args:
        nsys_args (list): The NSYS arguments as a list.

    Returns:
        tuple: A tuple containing the cleaned arguments list without the output option and the extracted
        output options as a list strings.
    """
    output_options = None
    output_option_index = None
    for i, arg in enumerate(nsys_args):
        if arg in ["--output", "-o"] and i + 1 < len(nsys_args):
            output_options = [arg, nsys_args[i + 1]]
            output_option_index = i
            break  # Stop after finding the first output option

    if output_option_index is not None:
        # Remove the output option and its value from the list
        cleaned_args = (
            nsys_args[:output_option_index] + nsys_args[output_option_index + 2 :]
        )
    else:
        cleaned_args = nsys_args

    return cleaned_args, output_options


def supports_output_option(nsys_args):
    """
    Checks if the command supports an output option.

    Args:
        nsys_args (list): The NSYS arguments as a list.

    Returns:
        bool: True if the command supports an output option, False otherwise.
    """
    OUTPUT_SUPPORTING_COMMANDS = [
        "analyze",
        "export",
        "profile",
        "start",
        "stats",
        "nvprof",
    ]
    return nsys_args[0] in OUTPUT_SUPPORTING_COMMANDS


def list_nsys_sessions(
    namespace, pod_name, container_name, nsys_path, nsys_session_prefix
):
    """
    Lists Nsight Systems sessions that match a given prefix.

    Args:
        namespace (str): The namespace of the pod.
        pod_name (str): The name of the pod.
        container_name (str): The name of the container.
        nsys_path (str): The path to the Nsight Systems executable.
        nsys_session_prefix (str): The prefix to filter the sessions.

    Returns:
        list: A list of Nsight Systems sessions that match the given prefix.
    """
    core_v1_api = client.CoreV1Api()
    session_list_cmd = f"{nsys_path} sessions list"
    try:
        exec_command = ["/bin/sh", "-c", session_list_cmd]
        response = stream.stream(
            core_v1_api.connect_get_namespaced_pod_exec,
            pod_name,
            namespace,
            container=container_name,
            command=exec_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
        )
        # Filter sessions based on the prefix
        sessions = [
            line.split()[4]
            for line in response.splitlines()
            if nsys_session_prefix in line
        ]
        return sessions
    except client.ApiException as e:
        print(f"An error occurred while listing NSYS sessions: {e}")
        return []


def copy_folder_from_pod(
    namespace, pod_name, source_folder, destination, remove_source=False
):
    """
    Copies .nsys-rep files from a folder inside a pod to a local destination.
    Optionally deletes .nsys-rep files in the source folder after copying.

    Args:
        namespace (str): The namespace of the pod.
        pod_name (str): The name of the pod.
        source_folder (str): The path of the folder inside the pod.
        destination (str): The destination path on the local machine.
        remove_source (bool): If True, delete all .nsys-rep files in the source folder after copying.

    Raises:
        subprocess.CalledProcessError: If there is an error during the operation.
    """
    find_command = (
        f"kubectl exec {pod_name} -n {namespace} -- sh -c 'find"
        f' {source_folder} -maxdepth 1 -name "*.nsys-rep"\''
    )
    try:
        result = subprocess.run(
            find_command, check=True, shell=True, text=True, capture_output=True
        )
        files = result.stdout.strip().split("\n")
        count_copied = 0
        for file in files:
            if file:
                local_path = file.replace(source_folder, destination, 1)
                copy_command = f"kubectl cp {namespace}/{pod_name}:{file} {local_path}"
                subprocess.run(copy_command, check=True, shell=True)
                count_copied += 1
        print(
            f"Profiling results folder with {count_copied} .nsys-rep files copied"
            " successfully."
        )
    except subprocess.CalledProcessError as e:
        print(f"Error copying .nsys-rep files: {e}")
        return

    if remove_source:
        # Delete all .nsys-rep files in the source folder after copying
        delete_command = (
            f"kubectl exec {pod_name} -n {namespace} -- sh -c 'rm"
            f" {source_folder}/*.nsys-rep'"
        )
        try:
            subprocess.run(delete_command, check=True, shell=True)
            print(".nsys-rep files deleted successfully from the pod.")
        except subprocess.CalledProcessError as e:
            print(f"Error deleting .nsys-rep files from the pod: {e}")


def copy_file_from_pod(namespace, pod_name, file, destination, remove_source=False):
    """
    Copies .nsys-rep files from a folder inside a pod to a local destination.
    Optionally deletes .nsys-rep files in the source folder after copying.

    Args:
        namespace (str): The namespace of the pod.
        pod_name (str): The name of the pod.
        file (str): The path of the file inside the pod.
        destination (str): The destination path on the local machine.
        remove_source (bool): If True, delete the file from the pod

    Raises:
        subprocess.CalledProcessError: If there is an error during the operation.
    """
    try:
        copy_command = (
            "kubectl cp"
            f" {namespace}/{pod_name}:{file} {destination}/{os.path.basename(file)}"
        )
        subprocess.run(copy_command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error copying {file} from the pod: {e}")
        return

    if remove_source:
        delete_command = f"kubectl exec {pod_name} -n {namespace} -- sh -c 'rm {file}'"
        try:
            subprocess.run(delete_command, check=True, shell=True)
            print(f"{file} deleted successfully from the pod.")
        except subprocess.CalledProcessError as e:
            print(f"Error deleting {file} from the pod: {e}")


def execute_download_command_for_pod(
    namespace,
    pod_name,
    container_name,
    nsys_path,
    destination,
    remove_source,
    sidecar_output_options,
    nsys_session_prefix,
):
    """
    Executes a command in a specific container of a pod and prints the output.

    Args:
        namespace (str): The namespace of the pod.
        pod_name (str): The name of the pod.
        container_name (str): The name of the container in the pod.
        nsys_path (str): The Nsight Systems path inside Pods.
        destination (str): Specifies the local directory to which profiling results will be downloaded.
        remove_source (bool): Delete results after copying.
        sidecar_output_options (List[str]): The output option from a Sidecar config file, as a list of strings.
        nsys_session_prefix (str): The prefix for Nsight Systems sessions running through the Sidecar.
    """
    env_vars = fetch_pod_env_vars(namespace, pod_name, container_name)

    sessions = list_nsys_sessions(
        namespace, pod_name, container_name, nsys_path, nsys_session_prefix
    )
    folders_to_copy = set()
    for session in sessions:
        nsys_output_option = sidecar_output_options
        if nsys_output_option:
            output_name = replace_placeholders(
                namespace,
                pod_name,
                container_name,
                env_vars,
                session,
                nsys_output_option[1],
            )
            folders_to_copy.add(os.path.dirname(output_name))

    for folder_to_copy in folders_to_copy:
        print(f"Copying folder: {folder_to_copy} from pod: {pod_name}")
        os.makedirs(destination, exist_ok=True)
        copy_folder_from_pod(
            namespace, pod_name, folder_to_copy, destination, remove_source
        )

    manager = DataStorageManager()
    output_options = manager.retrieve_output_options(
        namespace, pod_name, container_name
    )

    # copy only files that are not in the folders to copy
    normalized_folders_to_copy = [
        folder if folder.endswith("/") else folder + "/" for folder in folders_to_copy
    ]
    files_to_copy = [
        file
        for file in output_options
        if not any(file.startswith(folder) for folder in normalized_folders_to_copy)
    ]
    for file in files_to_copy:
        print(f"Copying file: {file} from pod: {pod_name}")
        os.makedirs(destination, exist_ok=True)
        copy_file_from_pod(namespace, pod_name, file, destination, remove_source)


def execute_command_for_pod(
    namespace,
    pod_name,
    container_name,
    nsys_path,
    nsys_args,
    sidecar_output_options,
    nsys_session_prefix,
):
    """
    Executes a command in a specific container of a pod and prints the output.

    Args:
        namespace (str): The namespace of the pod.
        pod_name (str): The name of the pod.
        container_name (str): The name of the container in the pod.
        nsys_path (str): The Nsight Systems path inside Pods.
        nsys_args (List[str]): The Nsight Systems arguments to execute, as a list of strings.
        sidecar_output_options (List[str]): The output option from a Sidecar config file, as a list of strings.
        nsys_session_prefix (str): The prefix for Nsight Systems sessions running through the Sidecar.
    """
    core_v1_api = client.CoreV1Api()

    env_vars = fetch_pod_env_vars(namespace, pod_name, container_name)

    sessions = list_nsys_sessions(
        namespace, pod_name, container_name, nsys_path, nsys_session_prefix
    )

    manager = DataStorageManager()
    for session in sessions:
        nsys_args_without_output, nsys_output_option = (
            extract_output_args_from_nsys_args(nsys_args)
        )
        if not nsys_output_option and supports_output_option(nsys_args):
            nsys_output_option = sidecar_output_options
        if nsys_output_option:
            nsys_output_option[1] = replace_placeholders(
                namespace,
                pod_name,
                container_name,
                env_vars,
                session,
                nsys_output_option[1],
            )
            manager.insert_output_option(
                namespace, pod_name, container_name, nsys_output_option[1]
            )

        command = (
            [nsys_path]
            + nsys_args_without_output
            + (nsys_output_option or [])
            + ["--session", session]
        )
        try:
            print("Executing command:", " ".join(shlex.quote(arg) for arg in command))
            response = stream.stream(
                core_v1_api.connect_get_namespaced_pod_exec,
                pod_name,
                namespace,
                container=container_name,
                command=command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
            )
            print(
                f"Output from pod {pod_name}, container {container_name}:\n{response}"
            )
        except client.ApiException as e:
            print(
                f"An error occurred while executing command in pod {pod_name},"
                f" container {container_name}: {e}"
            )


def check_pod_label_and_annotation(namespace, pod_name, sidecar_name):
    """
    Checks if the specified pod or its namespace has the 'sidecar-injector=enabled' label set,
    and verifies that the pod has the 'sidecar-injector.com/inject="yes"' annotation.

    The function prints a status message to stdout.

    Args:
        namespace (str): The namespace of the pod to check.
        pod_name (str): The name of the pod to check.
        sidecar_name (str): The name of the sidecar application.
    """
    core_v1_api = client.CoreV1Api()

    pod = core_v1_api.read_namespaced_pod(name=pod_name, namespace=namespace)
    ns = core_v1_api.read_namespace(name=namespace)

    pod_labels = pod.metadata.labels or {}
    ns_labels = ns.metadata.labels or {}
    annotations = pod.metadata.annotations or {}

    label_in_pod = pod_labels.get(sidecar_name) == "enabled"
    label_in_ns = ns_labels.get(sidecar_name) == "enabled"
    annotation_is_set = (
        annotations.get("sidecar-injector.nvidia.com/status") == "injected"
    )

    label_status = label_in_pod or label_in_ns

    if label_status and annotation_is_set:
        print(
            f"{Fore.GREEN}INJECTED: The NVIDIA Devtool Sidecar Injector injected into"
            f" pod '{pod_name}' in namespace '{namespace}'."
        )
        return

    status_message = f"{Fore.RED}NOT INJECTED: "

    if not label_status:
        status_message += (
            f"The target pod '{pod_name}' in namespace '{namespace}' does not have the"
            f" label '{sidecar_name}=enabled' set.\n"
        )
    else:
        status_message += (
            f"The injection did not occur for pod '{pod_name}' in namespace"
            f" '{namespace}'. Please, check that you restarted the pod after labeling"
            f" with '{sidecar_name}=enabled'."
        )
    print(status_message)


def main():
    colorama.init(autoreset=True)

    args = parse_arguments()
    config_data = get_helm_values("devtools-sidecar-injector")

    sidecar_name = config_data["appname"]
    label_selector = f"{sidecar_name}=enabled"
    binaries_dir = config_data["binariesDir"]
    nsys_relpath = config_data["profile"]["devtoolCmd"]
    config_nsys_args = config_data["profile"]["devtoolArgs"]
    nsys_session_prefix = config_data["profile"]["devtoolSessionPrefix"]
    nsys_path = os.path.join(binaries_dir, nsys_relpath)

    initialize_kubernetes_client()
    pods = get_unique_kubernetes_pods(
        args.field_selector, label_selector, label_selector
    )

    sidecar_output_options = extract_output_args_from_nsys_args_str(config_nsys_args)

    if args.command == "download":
        for pod in pods:
            if pod.status.phase != "Running":
                continue
            for container in pod.spec.containers:
                execute_download_command_for_pod(
                    pod.metadata.namespace,
                    pod.metadata.name,
                    container.name,
                    nsys_path,
                    args.destination,
                    args.remove_source,
                    sidecar_output_options,
                    nsys_session_prefix,
                )
    elif args.command == "nsys":
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    execute_command_for_pod,
                    pod.metadata.namespace,
                    pod.metadata.name,
                    container.name,
                    nsys_path,
                    args.nsys_args,
                    sidecar_output_options,
                    nsys_session_prefix,
                )
                for pod in pods
                if pod.status.phase == "Running"
                for container in pod.spec.containers
            ]

            for future in as_completed(futures):
                future.result()
    elif args.command == "check":
        check_pod_label_and_annotation(args.namespace, args.pod_name, sidecar_name)


if __name__ == "__main__":
    main()
