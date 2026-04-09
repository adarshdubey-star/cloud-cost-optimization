variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for instances"
  type        = string
  default     = "us-central1-a"
}

variable "machine_type" {
  description = "VM machine type for workload instances"
  type        = string
  default     = "e2-medium"
}

variable "min_instances" {
  description = "Minimum number of instances in MIG"
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum number of instances in MIG"
  type        = number
  default     = 10
}
