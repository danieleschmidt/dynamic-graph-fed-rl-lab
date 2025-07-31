# Project Configuration
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "dynamic-graph-fed-rl"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

# AWS Configuration
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "terraform_state_bucket" {
  description = "S3 bucket name for Terraform state"
  type        = string
}

variable "terraform_lock_table" {
  description = "DynamoDB table name for Terraform state locking"
  type        = string
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the EKS cluster endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict this in production
}

# EKS Configuration
variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

# General Node Group Configuration
variable "general_node_instance_types" {
  description = "Instance types for general node group"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge"]
}

variable "general_node_min_size" {
  description = "Minimum number of nodes in general node group"
  type        = number
  default     = 1
}

variable "general_node_max_size" {
  description = "Maximum number of nodes in general node group"
  type        = number
  default     = 10
}

variable "general_node_desired_size" {
  description = "Desired number of nodes in general node group"
  type        = number
  default     = 3
}

# GPU Node Group Configuration
variable "gpu_node_instance_types" {
  description = "Instance types for GPU node group"
  type        = list(string)
  default     = ["p3.2xlarge", "p3.8xlarge"]
}

variable "gpu_node_min_size" {
  description = "Minimum number of nodes in GPU node group"
  type        = number
  default     = 0
}

variable "gpu_node_max_size" {
  description = "Maximum number of nodes in GPU node group"
  type        = number
  default     = 5
}

variable "gpu_node_desired_size" {
  description = "Desired number of nodes in GPU node group"
  type        = number
  default     = 2
}

# CloudFormation Stack Name (for GPU nodes)
variable "cloudformation_stack_name" {
  description = "CloudFormation stack name for GPU nodes"
  type        = string
  default     = "eks-gpu-nodes"
}

# Monitoring Configuration
variable "enable_monitoring" {
  description = "Enable CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "Log retention in days"
  type        = number
  default     = 30
}

# Security Configuration
variable "enable_encryption" {
  description = "Enable EKS encryption at rest"
  type        = bool
  default     = true
}

variable "encryption_key_arn" {
  description = "KMS key ARN for EKS encryption"
  type        = string
  default     = null
}