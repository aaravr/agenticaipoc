package com.raghuet.service;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class CompletePayload {
   private String action;
   private String label;
   private String outcomeVariable;
}
