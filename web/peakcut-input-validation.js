/* Small, DOM-free validation helpers shared by PeakCut's composer and tests. */
(function exposePeakCutInputValidation(root) {
  root.PeakCutInputValidation = {
    validateOptions({ minLength, maxLength, startMinute, endMinute } = {}) {
      const min = Number(minLength);
      const max = Number(maxLength);
      if (Number.isFinite(min) && Number.isFinite(max) && min > max) {
        return { field: "o_maxlen", message: "Shortest clip length must be less than or equal to longest clip length." };
      }
      const hasStart = startMinute !== "" && startMinute != null;
      const hasEnd = endMinute !== "" && endMinute != null;
      if (hasEnd && Number(endMinute) <= (hasStart ? Number(startMinute) : 0)) {
        return { field: "o_to", message: "End minute must be after start minute." };
      }
      return null;
    },
  };
})(typeof window === "undefined" ? globalThis : window);
