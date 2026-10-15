// Typst markup headings serialize depth; explicit heading() calls serialize level.
export const headingLevel = heading => heading.level ?? heading.depth ?? 1;
