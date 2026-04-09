terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ============================================================
# Google Cloud Storage — model artifacts and workload data
# ============================================================
resource "google_storage_bucket" "ml_bucket" {
  name          = "${var.project_id}-cloud-cost-opt"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  lifecycle_rule {
    action { type = "Delete" }
    condition { age = 90 }
  }
}

# ============================================================
# VPC Network
# ============================================================
resource "google_compute_network" "vpc" {
  name                    = "cloud-opt-vpc"
  auto_create_subnetworks = true
}

resource "google_compute_firewall" "allow_http_ssh" {
  name    = "cloud-opt-allow-http-ssh"
  network = google_compute_network.vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22", "80", "8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["workload-vm"]
}

# ============================================================
# Instance Template — workload VMs with monitoring agent
# ============================================================
resource "google_compute_instance_template" "workload" {
  name_prefix  = "workload-vm-"
  machine_type = var.machine_type

  tags = ["workload-vm"]

  disk {
    source_image = "debian-cloud/debian-12"
    auto_delete  = true
    boot         = true
    disk_size_gb = 20
  }

  network_interface {
    network = google_compute_network.vpc.name
    access_config {}
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    # Install Ops Agent for Cloud Monitoring metrics
    curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
    sudo bash add-google-cloud-ops-agent-repo.sh --also-install

    # Install stress-ng for load generation
    sudo apt-get update && sudo apt-get install -y stress-ng python3 python3-pip nginx
    sudo systemctl start nginx

    echo "Workload VM ready" > /var/log/workload-init.log
  EOT

  service_account {
    scopes = ["cloud-platform"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ============================================================
# Managed Instance Group (MIG) — autoscaled VM fleet
# ============================================================
resource "google_compute_instance_group_manager" "workload_mig" {
  name               = "workload-mig"
  base_instance_name = "workload"
  zone               = var.zone

  version {
    instance_template = google_compute_instance_template.workload.self_link_unique
  }

  target_size = var.min_instances

  named_port {
    name = "http"
    port = 80
  }
}

# ============================================================
# GCP Autoscaler (baseline threshold-based — for comparison)
# ============================================================
resource "google_compute_autoscaler" "threshold_autoscaler" {
  name   = "threshold-autoscaler"
  zone   = var.zone
  target = google_compute_instance_group_manager.workload_mig.id

  autoscaling_policy {
    min_replicas    = var.min_instances
    max_replicas    = var.max_instances
    cooldown_period = 60

    cpu_utilization {
      target = 0.70
    }
  }
}

# ============================================================
# Cloud Monitoring — alert policy for SLA violations
# ============================================================
resource "google_monitoring_alert_policy" "high_cpu" {
  display_name = "High CPU Alert (SLA Violation)"
  combiner     = "OR"

  conditions {
    display_name = "CPU > 85% for 5 minutes"
    condition_threshold {
      filter          = "resource.type = \"gce_instance\" AND metric.type = \"compute.googleapis.com/instance/cpu/utilization\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0.85
      duration        = "300s"

      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = []
}

# ============================================================
# Outputs
# ============================================================
output "bucket_name" {
  value = google_storage_bucket.ml_bucket.name
}

output "mig_name" {
  value = google_compute_instance_group_manager.workload_mig.name
}

output "mig_zone" {
  value = var.zone
}

output "project_id" {
  value = var.project_id
}
